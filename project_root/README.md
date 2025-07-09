# PID Temperature Controller with Peltier Element and RTD Sensor

This project implements a bi-directional temperature controller using a Peltier element, an RTD temperature sensor, an Arduino microcontroller, and a programmable DC power supply. It features closed-loop PID control implemented in Python, with automatic tuning and setpoint correction, making it suitable for both experimental physics applications and open-source control systems.

## Project Structure

```
project_root/
├── TemperatureController.py # Main execution script integrating all components
├── main.py                 # Core control loop (called by TemperatureController)
├── FIT_PID_PARAMS.py       # Script to fit PID Ki/Kd vs. temperature
├── CORRECTED_SETPOINT.py   # Setpoint offset correction routine
└── arduino/
    └── RTD_and_DPDT.ino  # Arduino code for RTD reading and relay control
```

## Scripts Description

- **TemperatureController.py**: Main entry point. Executes the entire control system by importing and coordinating the logic from the other scripts.
- **main.py**: Main control loop logic. Reads temperature from Arduino, applies PID control, and sets the DP711 power supply output.
- **FIT_PID_PARAMS.py**: Fits Ki and Kd parameters as functions of temperature using polynomial interpolation based on experimental data.
- **CORRECTED_SETPOINT.py**: Applies empirical correction to user-defined temperature setpoints to compensate for steady-state offset.
- **arduino/TemperatureController.ino**: Arduino sketch for reading temperature from an RTD sensor and toggling a DPDT relay to control current direction in the Peltier.

## Requirements

- Python ≥ 3.8
- Required packages:
  ```bash
  pip install numpy matplotlib pyserial
  ```

## Hardware Used

- RTD temperature sensor (e.g. Pt100)
- Arduino (Uno or Nano recommended)
- DPDT relay (for polarity switching)
- Peltier element
- Rigol DP711 DC power supply
- RTD amplifier (e.g. MAX31865)

## How to Use

1. Upload the Arduino sketch in `arduino/` to your board.
2. Connect the DP711 to your computer via serial port.
3. Edit `TemperatureController.py` to configure serial ports and parameters.
4. Run the control loop:
   ```bash
   python TemperatureController.py
   ```
5. Adjust setpoint or PID parameters via GUI or script functions.

## Features

- Dual-mode PID control (separate parameters for heating and cooling)
- Polynomial-based parameter tuning and setpoint correction
- Asymmetric hysteresis to prevent relay chatter
- Real-time plotting and logging
- Safe voltage shutdown on exit

## Academic Use

This system was developed as part of a physics undergraduate thesis project. It is structured for clarity, reproducibility, and academic documentation. The code and setup are suitable for laboratory experiments involving temperature regulation, thermal response analysis, and feedback control systems.
