# -*- coding: utf-8 -*-
"""
DroneManager Module

This module provides the DroneManager class which controls a drone using AirSim 
and updates its visualization in CARLA. It supports multiple scenarios:

1. Tracking: The drone takes off, aligns with a vehicle (yaw = vehicle_yaw - 90) and tracks it.
2. Navigation: The drone directly navigates from an origin waypoint to a destination waypoint.
3. Monitoring: The drone takes off and hovers while monitoring its state.
  
Author: Keshu Wu <keshuw@tamu.edu>
License: TDG-Attribution-NonCommercial-NoDistrib
"""

import math
import os
import time
import cv2
import numpy as np
import carla
import airsim
import logging
from typing import Optional

# Configure logging if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= Utility Functions =================
def convert_carla_to_airsim(carla_location: carla.Location, hover_offset: float) -> airsim.Vector3r:
    """
    Convert a CARLA location (ENU) to an AirSim Vector3r (NED) with an added hover offset.
    
    Mapping:
      AirSim x = CARLA y
      AirSim y = -CARLA x   (to fix left/right inversion)
      AirSim z = - (CARLA z + hover_offset)
    
    Parameters:
        carla_location (carla.Location): Location in CARLA.
        hover_offset (float): Additional offset for z.
        
    Returns:
        airsim.Vector3r: The corresponding location in AirSim.
    """
    x = carla_location.y
    y = -carla_location.x
    z = -(carla_location.z + hover_offset)
    return airsim.Vector3r(x, y, z)

def convert_airsim_to_carla(airsim_vector: airsim.Vector3r) -> carla.Location:
    """
    Inverse conversion from AirSim (NED) to CARLA (ENU).
    
    Mapping:
      CARLA x = - AirSim y
      CARLA y = AirSim x
      CARLA z = - AirSim z
      
    Parameters:
        airsim_vector (airsim.Vector3r): The location in AirSim.
        
    Returns:
        carla.Location: The corresponding location in CARLA.
    """
    return carla.Location(x=-airsim_vector.y_val, y=airsim_vector.x_val, z=-airsim_vector.z_val)


def quaternion_to_euler(q) -> tuple:
    """
    Convert an AirSim quaternion to Euler angles (roll, pitch, yaw) in radians.
    
    Parameters:
        q: A quaternion with attributes w_val, x_val, y_val, z_val.
        
    Returns:
        A tuple (roll, pitch, yaw) in radians.
    """
    sinr_cosp = 2 * (q.w_val * q.x_val + q.y_val * q.z_val)
    cosr_cosp = 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q.w_val * q.y_val - q.z_val * q.x_val)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ================= DroneManager Class =================
class DroneManager:
    """
    Manages a drone in AirSim and its visualization in CARLA.
    
    Provides methods for takeoff/ascend, horizontal alignment, tracking,
    direct navigation, and monitoring.
    
    When tracking a vehicle, the CARLA drone visualizer’s yaw is set as:
      drone_yaw = vehicle_yaw - 90.
    """
    def __init__(self, 
                 airsim_client: airsim.MultirotorClient, 
                 drone_actor: carla.Actor, 
                 camera_sensor: carla.Actor, 
                 world: carla.World, 
                 update_interval: float, 
                 top_down_offset: float, 
                 drone_speed: float = 5):
        self.airsim_client = airsim_client
        self.drone_actor = drone_actor
        self.camera_sensor = camera_sensor
        self.world = world
        self.update_interval = update_interval
        self.top_down_offset = top_down_offset
        self.drone_speed = drone_speed
        self.latest_camera_image: Optional[np.ndarray] = None


    def _compute_dynamic_speed(self, vehicle_speed: float) -> float:
        """
        Compute a dynamic drone speed based on the vehicle's speed.
        """
        min_speed    = self.drone_speed * 0.6
        speed_margin = 1.0
        max_speed    = self.drone_speed * 2.0
        desired      = vehicle_speed + speed_margin
        return max(min_speed, min(desired, max_speed))

    def set_camera_callback(self, output_folder: str = "output") -> None:
        """Sets up the camera callback to update the latest image."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        def camera_callback(image: carla.SensorData) -> None:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            image_bgr = array[:, :, :3]
            # filename = os.path.join(output_folder, f"{image.frame:06d}.png")
            # cv2.imwrite(filename, image_bgr)
            self.latest_camera_image = image_bgr
        self.camera_sensor.listen(lambda image: camera_callback(image))

    def takeoff_and_ascend(self, takeoff_target: airsim.Vector3r, vehicle: Optional[carla.Actor] = None) -> None:
        """
        Commands the drone to take off and ascend to the target altitude.
        
        If a vehicle is provided, updates the visualizer using vehicle yaw; otherwise uses a default yaw of 0.
        """
        logging.info("Drone taking off in AirSim...")
        self.airsim_client.takeoffAsync().join()
        time.sleep(1)

        # If vehicle provided, use its speed to adjust takeoff speed
        if vehicle:
            vel = vehicle.get_velocity()
            vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            speed = self._compute_dynamic_speed(vehicle_speed)
        else:
            speed = self.drone_speed
        takeoff_future = self.airsim_client.moveToPositionAsync(
            takeoff_target.x_val, takeoff_target.y_val, takeoff_target.z_val, speed
        )

        logging.info("Ascending drone to target height...")
        while True:
            current_state = self.airsim_client.getMultirotorState().kinematics_estimated.position
            if abs(current_state.z_val - takeoff_target.z_val) < 1.0:
                break
            updated_location = convert_airsim_to_carla(current_state)
            if vehicle is not None:
                vehicle_yaw = vehicle.get_transform().rotation.yaw
                drone_yaw = vehicle_yaw - 90
            else:
                drone_yaw = 0
            self.drone_actor.set_transform(carla.Transform(
                updated_location,
                carla.Rotation(pitch=0, yaw=drone_yaw, roll=0)
            ))
            top_down_loc = carla.Location(
                x=updated_location.x,
                y=updated_location.y,
                z=updated_location.z + self.top_down_offset
            )
            self.world.get_spectator().set_transform(
                carla.Transform(top_down_loc, carla.Rotation(pitch=-90, yaw=90, roll=0))
            )
            if self.latest_camera_image is not None:
                cv2.imshow("Drone Camera View", self.latest_camera_image)
                cv2.waitKey(1)
            time.sleep(self.update_interval)
        takeoff_future.join()
        logging.info("Drone reached takeoff height.")

    def horizontal_alignment(self, target_airsim: airsim.Vector3r, vehicle: carla.Actor) -> None:
        """
        Aligns the drone horizontally with the target vehicle.
        """

        # Adjust speed to match vehicle’s current speed
        vel = vehicle.get_velocity()
        vehicle_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed = self._compute_dynamic_speed(vehicle_speed)
        horizontal_future = self.airsim_client.moveToPositionAsync(
            target_airsim.x_val, target_airsim.y_val, target_airsim.z_val, speed
        )
        while True:
            current_state = self.airsim_client.getMultirotorState().kinematics_estimated.position
            horizontal_distance = math.sqrt((current_state.x_val - target_airsim.x_val)**2 +
                                            (current_state.y_val - target_airsim.y_val)**2)
            vehicle_yaw = vehicle.get_transform().rotation.yaw
            updated_location = convert_airsim_to_carla(current_state)
            drone_yaw = vehicle_yaw - 90
            self.drone_actor.set_transform(carla.Transform(
                updated_location,
                carla.Rotation(pitch=0, yaw=drone_yaw, roll=0)
            ))
            top_down_loc = carla.Location(
                x=updated_location.x,
                y=updated_location.y,
                z=updated_location.z + self.top_down_offset
            )
            self.world.get_spectator().set_transform(
                carla.Transform(top_down_loc, carla.Rotation(pitch=-90, yaw=90, roll=0))
            )
            if self.latest_camera_image is not None:
                cv2.imshow("Drone Camera View", self.latest_camera_image)
                cv2.waitKey(1)
            if horizontal_distance < 1.0:
                break
            time.sleep(self.update_interval)
        horizontal_future.join()
        logging.info("Drone horizontally aligned with vehicle.")


    def tracking_loop(self, vehicle: carla.Actor, hover_offset: float) -> None:
        """
        Continuously tracks the target vehicle, dynamically adjusting drone speed
        to match vehicle velocity and distance.
        """
        logging.info("Starting tracking loop: Drone will hover above and track the vehicle.")
        try:
            carla_map = self.world.get_map()
            while True:
                # --- 1) Get vehicle state ---
                vehicle_transform = vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                vehicle_yaw = vehicle_transform.rotation.yaw
                velocity = vehicle.get_velocity()
                vehicle_speed = math.sqrt(
                    velocity.x**2 + velocity.y**2 + velocity.z**2
                )

                # --- 2) Dynamically choose drone speed ---
                if vehicle_speed  < 0.1:
                    # no move command → drone will hover at current position
                    commanded_speed = 0.0
                else:
                    commanded_speed = self._compute_dynamic_speed(vehicle_speed)

                # --- 3) Compute target and command drone ---
                target_airsim_pos = convert_carla_to_airsim(vehicle_location, hover_offset)
                airsim_drone_yaw = vehicle_yaw - 90
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=airsim_drone_yaw)
                self.airsim_client.moveToPositionAsync(
                    target_airsim_pos.x_val,
                    target_airsim_pos.y_val,
                    target_airsim_pos.z_val,
                    commanded_speed,
                    yaw_mode=yaw_mode
                )

                # --- 4) Read back actual state & update CARLA visualizer ---
                state = self.airsim_client.getMultirotorState().kinematics_estimated
                updated_location = convert_airsim_to_carla(state.position)
                roll, pitch, yaw_rad = quaternion_to_euler(state.orientation)
                updated_yaw = math.degrees(yaw_rad)

                self.drone_actor.set_transform(carla.Transform(
                    updated_location,
                    carla.Rotation(pitch=0, yaw=updated_yaw, roll=0)
                ))

                top_down_loc = carla.Location(
                    x=updated_location.x,
                    y=updated_location.y,
                    z=updated_location.z + self.top_down_offset
                )
                self.world.get_spectator().set_transform(
                    carla.Transform(top_down_loc, carla.Rotation(pitch=-90, yaw=90, roll=0))
                )

                # --- Optional: visualize upcoming waypoints ---
                current_wp = carla_map.get_waypoint(vehicle_location)
                waypoints = []
                temp_wp = current_wp
                for _ in range(20):
                    next_wps = temp_wp.next(2.0)
                    if not next_wps:
                        break
                    temp_wp = next_wps[0]
                    waypoints.append(temp_wp)
                for wp in waypoints:
                    self.world.debug.draw_point(
                        wp.transform.location,
                        size=0.1,
                        life_time=self.update_interval + 0.05,
                        persistent_lines=False,
                        color=carla.Color(0, 255, 0)
                    )

                logging.info("Vehicle: %s yaw=%.2f -> Drone Target (AirSim): %s with yaw=%.2f",
                             vehicle_location, vehicle_yaw, target_airsim_pos, airsim_drone_yaw)

                if self.latest_camera_image is not None:
                    cv2.imshow("Drone Camera View", self.latest_camera_image)
                    cv2.waitKey(1)

                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            logging.info("Tracking interrupted by user.")
        finally:
            logging.info("Landing drone in AirSim...")
            self.airsim_client.landAsync().join()
            self.airsim_client.armDisarm(False)
            self.airsim_client.enableApiControl(False)
            self.drone_actor.destroy()
            self.camera_sensor.destroy()
            cv2.destroyAllWindows()
            logging.info("Drone tracking simulation completed.")

    def direct_navigation(self, destination_airsim: airsim.Vector3r) -> None:
        """
        Directly navigates the drone from its current position to a destination.
        """
        logging.info("Starting navigation to destination: %s", destination_airsim)
        navigation_future = self.airsim_client.moveToPositionAsync(
            destination_airsim.x_val, destination_airsim.y_val, destination_airsim.z_val, self.drone_speed
        )
        while True:
            current_state = self.airsim_client.getMultirotorState().kinematics_estimated.position
            distance = math.sqrt((current_state.x_val - destination_airsim.x_val)**2 +
                                 (current_state.y_val - destination_airsim.y_val)**2 +
                                 (current_state.z_val - destination_airsim.z_val)**2)
            # updated_location = convert_airsim_to_carla(current_state)
            # self.drone_actor.set_transform(carla.Transform(
            #     updated_location, carla.Rotation(pitch=0, yaw=0, roll=0)
            # ))

            state = self.airsim_client.getMultirotorState().kinematics_estimated
            updated_location = convert_airsim_to_carla(state.position)
            roll, pitch, yaw_rad = quaternion_to_euler(state.orientation)
            updated_yaw = math.degrees(yaw_rad)

            self.drone_actor.set_transform(carla.Transform(
                updated_location,
                carla.Rotation(pitch=0, yaw=updated_yaw, roll=0)
            ))
            top_down_loc = carla.Location(
                x=updated_location.x, y=updated_location.y, z=updated_location.z + self.top_down_offset
            )
            self.world.get_spectator().set_transform(
                carla.Transform(top_down_loc, carla.Rotation(pitch=-90, yaw=90, roll=0))
            )
            if self.latest_camera_image is not None:
                cv2.imshow("Drone Camera View", self.latest_camera_image)
                cv2.waitKey(1)
            if distance < 1.0:
                break
            time.sleep(self.update_interval)
        navigation_future.join()
        logging.info("Drone reached destination.")

    def monitoring_loop(self) -> None:
        """
        Keeps the drone hovering and monitors its state.
        """
        logging.info("Starting monitoring loop: Drone will remain hovering.")
        try:
            while True:
                current_state = self.airsim_client.getMultirotorState().kinematics_estimated.position
                updated_location = convert_airsim_to_carla(current_state)
                self.drone_actor.set_transform(carla.Transform(
                    updated_location, carla.Rotation(pitch=0, yaw=0, roll=0)
                ))
                top_down_loc = carla.Location(
                    x=updated_location.x, y=updated_location.y, z=updated_location.z + self.top_down_offset
                )
                self.world.get_spectator().set_transform(
                    carla.Transform(top_down_loc, carla.Rotation(pitch=-90, yaw=90, roll=0))
                )
                if self.latest_camera_image is not None:
                    cv2.imshow("Drone Camera View", self.latest_camera_image)
                    cv2.waitKey(1)
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            logging.info("Monitoring interrupted by user.")
        finally:
            logging.info("Landing drone in AirSim...")
            self.airsim_client.landAsync().join()
            self.airsim_client.armDisarm(False)
            self.airsim_client.enableApiControl(False)
            self.drone_actor.destroy()
            self.camera_sensor.destroy()
            cv2.destroyAllWindows()
            logging.info("Drone monitoring simulation completed.")

    def run(self, vehicle: Optional[carla.Actor], takeoff_target: airsim.Vector3r, hover_offset: float,
            scenario: str = "tracking", destination_airsim: Optional[airsim.Vector3r] = None) -> None:
        """
        Executes the complete drone control procedure.
        
        Parameters:
            vehicle (Optional[carla.Actor]): The target vehicle (if applicable).
            takeoff_target (airsim.Vector3r): The AirSim takeoff target.
            hover_offset (float): Altitude offset for hovering.
            scenario (str): Scenario type ("tracking" (default), "navigation", or "monitoring").
            destination_airsim (Optional[airsim.Vector3r]): Destination for navigation.
        """
        if scenario == "tracking":
            self.takeoff_and_ascend(takeoff_target, vehicle)
            target_airsim = convert_carla_to_airsim(vehicle.get_transform().location, hover_offset)
            self.horizontal_alignment(target_airsim, vehicle)
            # Set vehicle autopilot.
            vehicle.set_autopilot(True)
            logging.info("Vehicle autopilot enabled. Vehicle is now moving.")
            self.tracking_loop(vehicle, hover_offset)
        elif scenario == "navigation":
            self.takeoff_and_ascend(takeoff_target, None)
            if destination_airsim is None:
                logging.error("Destination must be provided for navigation.")
                return
            self.direct_navigation(destination_airsim)
            logging.info("Landing drone after navigation...")
            self.airsim_client.landAsync().join()
        elif scenario == "monitoring":
            self.takeoff_and_ascend(takeoff_target, None)
            self.monitoring_loop()
        else:
            logging.error("Unknown scenario type: %s", scenario)
