# -*- coding: utf-8 -*-
"""
Scenario Manager for CARLA Simulation with Drone Control via AirSim.

This module manages CARLA simulation construction and provides multiple
drone scenarios (tracking a vehicle, direct navigation between waypoints, or
basic monitoring). The drone is controlled using AirSim and its state is
visualized in CARLA.
  
Author: Runsheng Xu <rxx3386@ucla.edu>
License: TDG-Attribution-NonCommercial-NoDistrib
"""

import math
import random
import sys
import os
import time
import cv2
import numpy as np
import carla
import airsim
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= Utility Functions =================
def convert_carla_to_airsim(carla_location: carla.Location, hover_offset: float) -> airsim.Vector3r:
    """
    Convert a CARLA location (ENU) to an AirSim Vector3r (NED) with an added hover offset.
    
    Mapping:
      AirSim x = CARLA y
      AirSim y = -CARLA x   (fixes left/right inversion)
      AirSim z = - (CARLA z + hover_offset)
    
    Parameters:
        carla_location (carla.Location): Location in CARLA (ENU).
        hover_offset (float): Additional offset added to the z-coordinate.
        
    Returns:
        airsim.Vector3r: The corresponding location in AirSim (NED).
    """
    x_airsim = carla_location.y
    y_airsim = -carla_location.x
    z_airsim = -(carla_location.z + hover_offset)
    return airsim.Vector3r(x_airsim, y_airsim, z_airsim)

def convert_airsim_to_carla(airsim_vector: airsim.Vector3r) -> carla.Location:
    """
    Convert an AirSim Vector3r (NED) to a CARLA Location (ENU).
    
    Inverse Mapping:
      CARLA x = - AirSim y
      CARLA y = AirSim x
      CARLA z = - AirSim z
      
    Parameters:
        airsim_vector (airsim.Vector3r): The location in AirSim.
        
    Returns:
        carla.Location: The corresponding location in CARLA.
    """
    return carla.Location(x=-airsim_vector.y_val, y=airsim_vector.x_val, z=-airsim_vector.z_val)

# ================= DroneManager Class =================
class DroneManager:
    """
    Manages a drone controlled via AirSim and visualized in CARLA.
    
    This class provides methods for the drone to take off, align horizontally,
    track a target vehicle, navigate directly, or simply hover and monitor its state.
    The CARLA drone visualizer’s yaw is set as: drone_yaw = vehicle_yaw - 90 when a vehicle is provided.
    """
    def __init__(self, 
                 airsim_client: airsim.MultirotorClient, 
                 drone_actor: carla.Actor, 
                 camera_sensor: carla.Actor, 
                 world: carla.World, 
                 update_interval: float, 
                 top_down_offset: float, 
                 drone_speed: float = 6):
        self.airsim_client: airsim.MultirotorClient = airsim_client
        self.drone_actor: carla.Actor = drone_actor
        self.camera_sensor: carla.Actor = camera_sensor
        self.world: carla.World = world
        self.update_interval: float = update_interval
        self.top_down_offset: float = top_down_offset
        self.drone_speed: float = drone_speed
        self.latest_camera_image: Optional[np.ndarray] = None

    def set_camera_callback(self, output_folder: str = "output") -> None:
        """
        Set up the camera callback to update the latest camera image.
        
        Parameters:
            output_folder (str): Directory to save captured images.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        def camera_callback(image: carla.SensorData):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            image_bgr = array[:, :, :3]
            filename = os.path.join(output_folder, f"{image.frame:06d}.png")
            cv2.imwrite(filename, image_bgr)
            self.latest_camera_image = image_bgr
        self.camera_sensor.listen(lambda image: camera_callback(image))

    def takeoff_and_ascend(self, takeoff_target: airsim.Vector3r, vehicle: Optional[carla.Actor] = None) -> None:
        """
        Commands the drone to take off and ascend to the target altitude.
        
        If a vehicle is provided, the CARLA visualizer's yaw is updated based on the vehicle's yaw.
        Otherwise, a default yaw of 0 is used.
        
        Parameters:
            takeoff_target (airsim.Vector3r): The target position for takeoff.
            vehicle (Optional[carla.Actor]): The target vehicle (if any).
        """
        logging.info("Drone taking off in AirSim...")
        self.airsim_client.takeoffAsync().join()
        time.sleep(1)
        takeoff_future = self.airsim_client.moveToPositionAsync(
            takeoff_target.x_val, takeoff_target.y_val, takeoff_target.z_val, self.drone_speed
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
        Align the drone horizontally with the target vehicle.
        
        Parameters:
            target_airsim (airsim.Vector3r): The target horizontal position (with hover offset).
            vehicle (carla.Actor): The vehicle to align with.
        """
        horizontal_future = self.airsim_client.moveToPositionAsync(
            target_airsim.x_val, target_airsim.y_val, target_airsim.z_val, self.drone_speed
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
        Continuously track the target vehicle.
        
        Parameters:
            vehicle (carla.Actor): The vehicle to track.
            hover_offset (float): Altitude offset for hovering.
        """
        logging.info("Starting tracking loop: Drone will hover above and track the vehicle.")
        try:
            carla_map = self.world.get_map()
            while True:
                vehicle_transform = vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                vehicle_yaw = vehicle_transform.rotation.yaw

                target_airsim_pos = convert_carla_to_airsim(vehicle_location, hover_offset)
                airsim_drone_yaw = vehicle_yaw - 90
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=airsim_drone_yaw)
                self.airsim_client.moveToPositionAsync(
                    target_airsim_pos.x_val,
                    target_airsim_pos.y_val,
                    target_airsim_pos.z_val,
                    self.drone_speed,
                    yaw_mode=yaw_mode
                )

                updated_location = convert_airsim_to_carla(target_airsim_pos)
                self.drone_actor.set_transform(carla.Transform(
                    updated_location,
                    carla.Rotation(pitch=0, yaw=vehicle_yaw - 90, roll=0)
                ))
                top_down_loc = carla.Location(
                    x=updated_location.x,
                    y=updated_location.y,
                    z=updated_location.z + self.top_down_offset
                )
                self.world.get_spectator().set_transform(
                    carla.Transform(top_down_loc, carla.Rotation(pitch=-90, yaw=90, roll=0))
                )
                # Visualize upcoming waypoints.
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
                        wp.transform.location, size=0.1, life_time=self.update_interval + 0.05,
                        persistent_lines=False, color=carla.Color(0, 255, 0)
                    )
                logging.info("Vehicle: %s yaw=%.2f -> Drone Target (AirSim): %s with yaw=%.2f",
                             vehicle_location, vehicle_yaw,
                             target_airsim_pos, airsim_drone_yaw)
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
        Directly navigate the drone from its current position to a destination.
        Updates the CARLA drone visualizer based on AirSim's dynamics.
        
        Parameters:
            destination_airsim (airsim.Vector3r): The destination position in AirSim coordinates.
        """
        logging.info("Starting direct navigation to destination: %s", destination_airsim)
        navigation_future = self.airsim_client.moveToPositionAsync(
            destination_airsim.x_val, destination_airsim.y_val, destination_airsim.z_val, self.drone_speed
        )
        while True:
            current_state = self.airsim_client.getMultirotorState().kinematics_estimated.position
            distance = math.sqrt((current_state.x_val - destination_airsim.x_val)**2 +
                                 (current_state.y_val - destination_airsim.y_val)**2 +
                                 (current_state.z_val - destination_airsim.z_val)**2)
            updated_location = convert_airsim_to_carla(current_state)
            self.drone_actor.set_transform(carla.Transform(
                updated_location,
                carla.Rotation(pitch=0, yaw=0, roll=0)
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
            if distance < 1.0:
                break
            time.sleep(self.update_interval)
        navigation_future.join()
        logging.info("Drone reached destination.")

    def monitoring_loop(self) -> None:
        """
        After takeoff, keep the drone hovering and monitor its state.
        """
        logging.info("Starting monitoring loop: Drone will remain hovering and monitor its state.")
        try:
            while True:
                current_state = self.airsim_client.getMultirotorState().kinematics_estimated.position
                updated_location = convert_airsim_to_carla(current_state)
                self.drone_actor.set_transform(carla.Transform(
                    updated_location,
                    carla.Rotation(pitch=0, yaw=0, roll=0)
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
        Execute the complete drone control procedure.
        
        For "tracking", the drone takes off, aligns with the vehicle, and tracks it.
        For "direct_navigation", no vehicle is used; the drone navigates from its origin to a destination.
        For "monitoring", the drone takes off and then hovers while monitoring its state.
        
        Parameters:
            vehicle (Optional[carla.Actor]): The target vehicle (if applicable).
            takeoff_target (airsim.Vector3r): The AirSim target for takeoff.
            hover_offset (float): The altitude offset for hovering.
            scenario (str): The scenario type ("tracking", "direct_navigation", or "monitoring").
            destination_airsim (Optional[airsim.Vector3r]): Destination for direct navigation.
        """
        if scenario == "tracking":
            self.takeoff_and_ascend(takeoff_target, vehicle)
            target_airsim = convert_carla_to_airsim(vehicle.get_transform().location, hover_offset)
            self.horizontal_alignment(target_airsim, vehicle)
            # Enable vehicle autopilot since tracking uses a vehicle.
            vehicle.set_autopilot(True)
            logging.info("Vehicle autopilot enabled. Vehicle is now moving.")
            self.tracking_loop(vehicle, hover_offset)
        elif scenario == "direct_navigation":
            self.takeoff_and_ascend(takeoff_target, None)
            if destination_airsim is None:
                logging.error("Destination must be provided for direct navigation.")
                return
            self.direct_navigation(destination_airsim)
            logging.info("Landing drone after direct navigation...")
            self.airsim_client.landAsync().join()
        elif scenario == "monitoring":
            self.takeoff_and_ascend(takeoff_target, None)
            self.monitoring_loop()
        else:
            logging.error("Unknown scenario type: %s", scenario)

# ================= ScenarioManager Class =================
class ScenarioManager:
    """
    Manages the overall CARLA simulation.
    This class sets up the world, spawns a vehicle (if needed), and uses DroneManager to control the drone.
    It supports multiple scenarios via the scenario_params dictionary.
    
    scenario_params should include a key "scenario_type" with one of the following values:
      - "tracking" for vehicle tracking,
      - "direct_navigation" for navigating between two waypoints,
      - "monitoring" for basic hover and monitoring.
    """
    def __init__(self, scenario_params: dict, apply_ml: bool, carla_version: str,
                 xodr_path: Optional[str] = None, town: Optional[str] = None,
                 cav_world: Optional[object] = None, comm_manager: Optional[object] = None) -> None:
        self.scenario_params: dict = scenario_params
        self.carla_version: str = carla_version

        simulation_config = scenario_params['world']
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])

        self.client: carla.Client = carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)

        # Initialize AirSim client for drone control.
        self.airsim_client: airsim.MultirotorClient = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.enableApiControl(True)
        self.airsim_client.armDisarm(True)

        if xodr_path:
            # Assume load_customized_world is defined elsewhere.
            self.world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                self.world = self.client.load_world(town)
            except RuntimeError:
                sys.exit(f"Town {town} is not found in your CARLA repo!")
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        # self.origin_settings = self.world.get_settings()
        # new_settings = self.world.get_settings()
        # if simulation_config['sync_mode']:
        #     new_settings.synchronous_mode = True
        #     new_settings.fixed_delta_seconds = simulation_config['fixed_delta_seconds']
        # else:
        #     sys.exit('ERROR: Only synchronous simulation mode is supported.')
        # self.world.apply_settings(new_settings)

        if 'weather' in simulation_config:
            self.world.set_weather(carla.WeatherParameters(**simulation_config['weather']))

        self.carla_map: carla.Map = self.world.get_map()
        self.apply_ml: bool = apply_ml
        self.comm_manager = comm_manager
        self.cav_world = cav_world

        # Drone parameters.
        self.DRONE_TAKEOFF_HEIGHT: float = 60  # m
        self.DRONE_HOVER_OFFSET: float = self.DRONE_TAKEOFF_HEIGHT
        self.DRONE_SPEED: float = 6           # m/s
        self.DRONE_GROUND_SHIFT: float = 5.0    # m

        self.TOP_DOWN_OFFSET: float = 5.0       # m
        self.UPDATE_INTERVAL: float = 0.033     # seconds

    def run_drone_tracking(self) -> None:
        """
        Based on scenario_params, run the desired drone scenario.
        
        "tracking": spawn a vehicle and track it.
        "direct_navigation": pick two random waypoints as origin and destination and navigate.
        "monitoring": spawn the drone and have it hover while monitoring its state.
        """
        scenario_type = self.scenario_params.get("scenario_type", "tracking")
        blueprint_library = self.world.get_blueprint_library()

        if scenario_type == "tracking":
            # Spawn a vehicle.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available in the CARLA map!")
            vehicle_spawn_point = random.choice(spawn_points)
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)
            print(f"Spawned CARLA vehicle at spawn point: {vehicle_spawn_point}")

            # Compute nearby drone spawn location.
            drone_spawn_location = vehicle_spawn_point.location - carla.Location(
                x=self.DRONE_GROUND_SHIFT, y=self.DRONE_GROUND_SHIFT, z=0)

            # Initialize drone visualizer and AirSim pose.
            drone_actor, camera_sensor = self.initialize_drone(blueprint_library, drone_spawn_location)

            # Create DroneManager instance.
            drone_manager = DroneManager(self.airsim_client, drone_actor, camera_sensor, self.world,
                                         self.UPDATE_INTERVAL, self.TOP_DOWN_OFFSET, drone_speed=self.DRONE_SPEED)
            drone_manager.set_camera_callback()

            # Set spectator view.
            initial_loc = drone_actor.get_transform().location
            spectator = self.world.get_spectator()
            top_down_location = carla.Location(x=initial_loc.x, y=initial_loc.y, z=initial_loc.z + self.TOP_DOWN_OFFSET)
            top_down_rotation = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator.set_transform(carla.Transform(top_down_location, top_down_rotation))
            print("Set CARLA spectator to top-down view.")

            # Define takeoff target using the vehicle's spawn point.
            takeoff_target = airsim.Vector3r(
                convert_carla_to_airsim(vehicle_spawn_point.location, 0).x_val,
                convert_carla_to_airsim(vehicle_spawn_point.location, 0).y_val,
                -self.DRONE_TAKEOFF_HEIGHT
            )

            # Run the tracking scenario.
            drone_manager.run(vehicle, takeoff_target, self.DRONE_HOVER_OFFSET, scenario="tracking")

        elif scenario_type == "direct_navigation":
            # For direct navigation, do not spawn any vehicle.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available in the CARLA map!")
            origin_transform, destination_transform = random.sample(spawn_points, 2)
            print(f"Selected origin: {origin_transform} and destination: {destination_transform}")

            # Spawn drone at the origin.
            drone_actor, camera_sensor = self.initialize_drone(blueprint_library, origin_transform.location)

            # Create DroneManager instance.
            drone_manager = DroneManager(self.airsim_client, drone_actor, camera_sensor, self.world,
                                         self.UPDATE_INTERVAL, self.TOP_DOWN_OFFSET, drone_speed=self.DRONE_SPEED)
            drone_manager.set_camera_callback()

            # Set spectator view.
            initial_loc = drone_actor.get_transform().location
            spectator = self.world.get_spectator()
            top_down_location = carla.Location(x=initial_loc.x, y=initial_loc.y, z=initial_loc.z + self.TOP_DOWN_OFFSET)
            top_down_rotation = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator.set_transform(carla.Transform(top_down_location, top_down_rotation))
            print("Set CARLA spectator to top-down view.")

            # Define takeoff target using the origin's location.
            takeoff_target = airsim.Vector3r(
                convert_carla_to_airsim(origin_transform.location, 0).x_val,
                convert_carla_to_airsim(origin_transform.location, 0).y_val,
                -self.DRONE_TAKEOFF_HEIGHT
            )

            # Command takeoff and ascend, then direct navigation to destination.
            drone_manager.takeoff_and_ascend(takeoff_target, None)
            destination_airsim = convert_carla_to_airsim(destination_transform.location, self.DRONE_HOVER_OFFSET)
            drone_manager.direct_navigation(destination_airsim)
            logging.info("Landing drone after direct navigation...")
            self.airsim_client.landAsync().join()

        elif scenario_type == "monitoring":
            # For monitoring, do not spawn any vehicle.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available in the CARLA map!")
            monitor_spawn_point = random.choice(spawn_points)
            print(f"Selected monitoring spawn point: {monitor_spawn_point}")

            # Spawn drone at the monitoring location.
            drone_actor, camera_sensor = self.initialize_drone(blueprint_library, monitor_spawn_point.location)

            # Create DroneManager instance.
            drone_manager = DroneManager(self.airsim_client, drone_actor, camera_sensor, self.world,
                                         self.UPDATE_INTERVAL, self.TOP_DOWN_OFFSET, drone_speed=self.DRONE_SPEED)
            drone_manager.set_camera_callback()

            # Set spectator view.
            initial_loc = drone_actor.get_transform().location
            spectator = self.world.get_spectator()
            top_down_location = carla.Location(x=initial_loc.x, y=initial_loc.y, z=initial_loc.z + self.TOP_DOWN_OFFSET)
            top_down_rotation = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator.set_transform(carla.Transform(top_down_location, top_down_rotation))
            print("Set CARLA spectator to top-down view.")

            # Define takeoff target using the monitoring spawn point.
            takeoff_target = airsim.Vector3r(
                convert_carla_to_airsim(monitor_spawn_point.location, 0).x_val,
                convert_carla_to_airsim(monitor_spawn_point.location, 0).y_val,
                -self.DRONE_TAKEOFF_HEIGHT
            )

            # Run the monitoring scenario: takeoff then hover.
            drone_manager.run(vehicle=None, takeoff_target=takeoff_target, hover_offset=self.DRONE_HOVER_OFFSET, scenario="monitoring")
        else:
            print(f"Unknown scenario type: {scenario_type}")

    def initialize_drone(self, blueprint_library: carla.BlueprintLibrary, drone_spawn_location: carla.Location) -> tuple:
        """
        Initialize the drone visualizer and AirSim pose.
        
        Parameters:
            blueprint_library (carla.BlueprintLibrary): The CARLA blueprint library.
            drone_spawn_location (carla.Location): The spawn location for the drone.
        
        Returns:
            tuple: (drone_actor, camera_sensor)
        """
        return self.initialize_drone_on_ground(self.airsim_client, blueprint_library, drone_spawn_location)

    def initialize_drone_on_ground(self, airsim_client: airsim.MultirotorClient, blueprint_library: carla.BlueprintLibrary,
                                   drone_spawn_location: carla.Location) -> tuple:
        """
        Spawn the CARLA drone visualizer and set the AirSim pose.
        
        Parameters:
            airsim_client (airsim.MultirotorClient): The AirSim client.
            blueprint_library (carla.BlueprintLibrary): The CARLA blueprint library.
            drone_spawn_location (carla.Location): The spawn location for the drone.
        
        Returns:
            tuple: (drone_actor, camera_sensor)
        """
        drone_transform = carla.Transform(drone_spawn_location, carla.Rotation(pitch=0, yaw=0, roll=0))
        drone_blueprint = blueprint_library.find('static.prop.shoppingcart')
        drone_actor = self.world.spawn_actor(drone_blueprint, drone_transform)
        print(f"Spawned CARLA drone visualizer at location: {drone_spawn_location}")

        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=-1),
            carla.Rotation(pitch=-90, yaw=90, roll=0)
        )
        camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=drone_actor)
        print("Attached camera sensor to drone visualizer.")

        airsim_ground_position = convert_carla_to_airsim(drone_spawn_location, hover_offset=0)
        drone_initial_orientation = airsim.to_quaternion(0, 0, 0)
        drone_initial_pose = airsim.Pose(airsim_ground_position, drone_initial_orientation)
        airsim_client.simSetVehiclePose(drone_initial_pose, True)
        print(f"Initialized AirSim drone at position: {airsim_ground_position}")

        return drone_actor, camera_sensor

    def destroy_actors(self) -> None:
        """
        Destroy all actors in the CARLA world.
        """
        for actor in self.world.get_actors():
            actor.destroy()

    def close(self) -> None:
        """
        Restore original world settings.
        """
        self.world.apply_settings(self.origin_settings)


if __name__ == "__main__":
    scenario_params = {
        'world': {
            'client_port': 2000,
            'sync_mode': False,
            'fixed_delta_seconds': 0.033,
            'weather': {
                'sun_altitude_angle': 70,
                'cloudiness': 10,
                'precipitation': 0,
                'precipitation_deposits': 0,
                'wind_intensity': 0,
                'fog_density': 0,
                'fog_distance': 0,
                'fog_falloff': 0,
                'wetness': 0
            },
            'seed': 42
        },
        # Set "scenario_type" to "tracking", "direct_navigation", or "monitoring"
        "scenario_type": "tracking"
    }
    sm = ScenarioManager(scenario_params, apply_ml=False, carla_version='0.9.15')
    sm.run_drone_tracking()
