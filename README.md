# AirSim CARLA Co-Simulation

## Overview

This framework combines CARLA and AirSim to control a drone within a CARLA simulation. It supports three distinct operational modes:

1. **Tracking (default):**
   - A vehicle is spawned at a random location.
   - A drone is spawned nearby, takes off, aligns its heading (using `yaw = vehicle_yaw - 90`), and continuously follows the vehicle.
   - Once the drone is aligned, the vehicle is set to autopilot.

2. **Navigation:**
   - Two random waypoints on the CARLA map are selected as the drone’s origin and destination.
   - The drone spawns at the origin, takes off, and then directly navigates to the destination.

3. **Monitoring:**
   - The drone is spawned at a random location, takes off, and hovers while continuously updating its state and camera view.

## File Structure

- **drone_manager.py**  
  Contains the `DroneManager` class and utility functions for converting between CARLA (ENU) and AirSim (NED) coordinate systems. It provides methods for:
  - Drone takeoff/ascend
  - Horizontal alignment
  - Tracking a vehicle
  - Direct navigation
  - Monitoring (hovering)

- **scenario_manager.py**  
  Contains the `ScenarioManager` class, which sets up the CARLA world, spawns the necessary actors (vehicle and/or drone), and selects the desired scenario based on the `scenario_type` parameter in the configuration.

## Usage

1. **Start the Simulators:**  
   Before running the code, launch the CARLA Simulator by starting `CarlaUE4.exe` and launch the AirSim environment by starting `Blocks.exe`.

2. **Configure the Simulation:**  
   Edit the `scenario_params` dictionary in `scenario_manager.py` to adjust the simulation parameters. Set the `"scenario_type"` key to one of the following values:
   - `"tracking"` (default)
   - `"navigation"`
   - `"monitoring"`

3. **Run the Simulation:**  
   Execute the scenario manager script from the command line:
   ```bash
   python scenario_manager.py
## Requirements

- **CARLA Simulator:** Ensure that the CARLA Simulator is running.

- **AirSim Binaries:**  
  Download the binaries (e.g., *Blocks.zip*) from the [AirSim Releases](https://github.com/Microsoft/AirSim/releases) page rather than building from source.

- **AirSim Source Code:**  
  Download the "Source code (zip)" from the [AirSim Releases](https://github.com/Microsoft/AirSim/releases) page and place the `\AirSim-1.8.1-windows\PythonClient\airsim` folder in the same directory.

- **Python 3.x** and required libraries:  
  `carla`, `airsim`, `numpy`, `opencv-python`, etc.
