# -*- coding: utf-8 -*-
"""

@author: Diego md



"""

import time
import serial
import matplotlib.pyplot as plt
import numpy as np
import atexit
from matplotlib.widgets import Button, TextBox
from collections import deque

from FIT_PID_PARAMS import get_ki_cool, get_kd_cool, get_ki_heat, get_kd_heat
from CORRECTED_SETPOINT import corrected_setpoint


class TemperatureController:
    def __init__(self, set_temperature,
                 # PID parameters in heating mode
                 Kp_heat=0.65, Ki_heat=0.0, Kd_heat=0.0,
                 # PID parameters in cooling mode
                 Kp_cool=0.75, Ki_cool=0.0, Kd_cool=0.00,
                 # Other parameters
                 power_supply_port='COM1', arduino_port='COM5',
                 baud_rate_power_supply=9600, baud_rate_arduino=115200,
                 high_voltage=7.0, low_voltage=0.0, 
                 integral_window=10,
                 # Power factor control
                 heating_power_factor=0.75, cooling_power_factor=1.0,
                 # Asymmetric hysteresis thresholds
                 heating_threshold=-2.0,  # °C below setpoint
                 cooling_threshold=2.0,   # °C above setpoint
                 show_threshold_lines=False,   
                 enable_extra_plots=False):
                    #Enables threshold lines and extra window for voltage plots

        # Basic parameters
        
        self.power_supply_port = power_supply_port
        self.arduino_port = arduino_port
        self.baud_rate_power_supply = baud_rate_power_supply
        self.baud_rate_arduino = baud_rate_arduino
        self.high_voltage = high_voltage
        self.low_voltage = low_voltage

        # Variables for offset correction
        self.original_set_temperature = set_temperature
        self.set_temperature = corrected_setpoint(set_temperature)
        
        # Temperature range
        self.min_temperature = 0.0
        self.max_temperature = 55.0

        # Dual PID parameters
        self.Kp_heat = Kp_heat
        self.Ki_heat = Ki_heat
        self.Kd_heat = Kd_heat
        self.Kp_cool = Kp_cool
        self.Ki_cool = Ki_cool
        self.Kd_cool = Kd_cool

        # Power factor parameters
        self.heating_power_factor = heating_power_factor
        self.cooling_power_factor = cooling_power_factor
        
        # Asimetric hysteresis parameters (±2°C)
        self.heating_threshold = heating_threshold
        self.cooling_threshold = cooling_threshold

        # State variables
        self.start_time = None
        self.temperature_values = []
        self.ser1 = None
        self.ser2 = None
        self.stop_flag = False
        self.relay_state = None            # 'ON' if heating, 'OFF' if cooling
        self.shutdown_complete = False

        # History for PID calculations
        self.error_history = deque(maxlen=integral_window)
        self.error_integral_heat = 0.0
        self.error_integral_cool = 0.0
        self.last_error = 0.0

        # Anti-windup limits
        self.integral_limit_heat = 10.0
        self.integral_limit_cool = 10.0

        # Other variables 
        self.temp_history = deque(maxlen=30)
        self.current_power = 0.0
        self.last_power_update_time = 0
        self.heating_mode = True       # True = heating; False = cooling mode
        self.voltage_values = []                 # List to save voltage values
        self.show_threshold_lines = show_threshold_lines
        self.enable_extra_plots = enable_extra_plots
        self.setpoint_history = []                     # List of (time, value)
        
        # Cleanup function 
        atexit.register(self.emergency_shutdown)

    def setup_serial_ports(self):
        """Serial ports for power supply and Arduino"""
        try:
            self.ser1 = serial.Serial(port=self.power_supply_port,
                                      baudrate=self.baud_rate_power_supply,
                                      timeout=0.5)  # Increased timeout to improve robustness

            self.ser2 = serial.Serial(port=self.arduino_port,
                                      baudrate=self.baud_rate_arduino,
                                      timeout=0.5)  # 
            
            # Initial command to check response
            success = self.send_command(self.ser1, ":VOLT:LEV 0")
            if success:
                print("Power supply / Arduino connection established")
                return True
            else:
                print("Error sending command")
                return False
        except Exception as e:
            print(f"Error in serial port: {e}")
            return False

    def send_command(self, serial_port, command):
        """Function to send serial commands with robust error management"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Clean buffer before sending command
                serial_port.reset_input_buffer()

                # Ensure correct format
                formatted_command = command.strip() + '\r\n'
                serial_port.write(formatted_command.encode())

                time.sleep(0.1)

                # Check for errors
                if serial_port.in_waiting > 0:
                    response = serial_port.readline().decode().strip()
                    if "Incorrect" in response or "Error" in response:
                        print(
                            f"Rejected command (attempt {attempt+1}/{max_attempts}): {response}")
                        time.sleep(0.2)
                        continue

                return True
            except Exception as e:
                print(
                    f"Error sending command (attempt {attempt+1}/{max_attempts}): {e}")
                time.sleep(0.2)

        return False

    def set_voltage_safely(self, voltage):
        """Safely set voltage in the power supply"""
        # Clip voltage to the allowed range
        safe_voltage = np.clip(voltage, 0, self.high_voltage)

        # Formmating issues prevention
        command = f":VOLT:LEV {safe_voltage:.3f}"

        # Send command with retries
        success = self.send_command(self.ser1, command)

        # Check 
        if not success and not self.stop_flag:
            # Alternative SCPI command format
            alt_command = f":VOLTAGE:LEVEL {safe_voltage:.3f}"
            self.send_command(self.ser1, alt_command)

        return success

    def calculate_temp_rate(self):
        """Calculates temperature change rate °C/second"""
        if len(self.temp_history) >= 5:
            # Use last 5 seconds
            recent_temps = list(self.temp_history)[-5:]
            temp_rate = (recent_temps[-1] - recent_temps[0]) / 5  # °C/sec
            return temp_rate
        return 0

    def determine_relay_state(self, temp, setpoint):
        """Determines relay state based on asymmetric hysteresis with wide thresholds(±2°C)"""
        error = temp - setpoint

        # Establish relay state based on error
        if self.relay_state is None:
            self.relay_state = 'ON' if error < 0 else 'OFF'
            return self.relay_state, True

        # Check relay state based on threshold
        if error <= self.heating_threshold and self.relay_state == 'OFF':
            print(f"Switching to HEATING mode (error: {error:.4f}°C)")
            return 'ON', True
        elif error >= self.cooling_threshold and self.relay_state == 'ON':
            print(f"Switching to COOLING mode (error: {error:.4f}°C)")
            return 'OFF', True

        # Mantain current state
        return self.relay_state, False

    def calculate_pid_output(self, temp, error):
        """Calculate PID output in separated modes for heating and cooling"""

        self.temp_history.append(temp)

        self.heating_mode = self.relay_state == 'ON'  # 'ON' = heating

        # Select dual PID parameters
        if self.heating_mode:
            Kp = self.Kp_heat
            Ki = self.Ki_heat
            Kd = self.Kd_heat
            power_factor = self.heating_power_factor
            error_integral = self.error_integral_heat
            integral_limit = self.integral_limit_heat
        else:
            Kp = self.Kp_cool
            Ki = self.Ki_cool
            Kd = self.Kd_cool
            power_factor = self.cooling_power_factor
            error_integral = self.error_integral_cool
            integral_limit = self.integral_limit_cool

        # Update error history
        self.error_history.append(error)

        # Integral term (including anti-windup)
        error_integral = sum(self.error_history)
        error_integral = np.clip(
            error_integral, -integral_limit, integral_limit)

        # Update integral 
        if self.heating_mode:
            self.error_integral_heat = error_integral
        else:
            self.error_integral_cool = error_integral

        # Derivative term
        error_derivative = error - self.last_error
        self.last_error = error

        # Calculate PID output
        output = abs(Kp*error + Ki*error_integral + Kd*error_derivative)

        # Set power factor
        output = output * power_factor

        # Apply damping based on the rate of temperature change
        temp_rate = self.calculate_temp_rate()
        if self.heating_mode and error < -5.0:
            pass
        elif self.heating_mode and temp_rate > 0.01:
            #Reduce power factor to allow more heating power
            damping = max(0.5, 1.0 - temp_rate * 5.0)
            output *= damping


        output_voltage = np.clip(output, self.low_voltage, self.high_voltage)

        return output_voltage

    def stop_simulation(self, event):
        """Stops simulation when pressing the button"""
        self.stop_flag = True
        self.emergency_shutdown()

    def emergency_shutdown(self):
        """Emergency shutdown establishing voltage to zero"""
        if self.shutdown_complete:
            return
        try:
            if self.ser1 is not None and self.ser1.is_open:
                # try different commands to ensure zero voltage
                self.set_voltage_safely(0.0)
                time.sleep(0.1)
                self.send_command(self.ser1, ":VOLT 0")
                time.sleep(0.1)
                self.send_command(self.ser1, ":VOLTAGE 0")
                time.sleep(0.1)
                self.send_command(self.ser1, ":SOURCE:VOLTAGE 0")

                print("Voltage OFF")
        except Exception as e:
            print(f"{e}")
        finally:
            self.shutdown_complete = True

    def cleanup(self):
        """Cleanup and close connections"""
        try:
            self.emergency_shutdown()

            # Close serial ports
            if self.ser1 is not None and self.ser1.is_open:
                self.ser1.close()
            if self.ser2 is not None and self.ser2.is_open:
                self.ser2.close()
        except Exception as e:
            print(f"Error durante limpieza: {e}")

        execution_time = time.time() - self.start_time
        print("Execution time: ", execution_time, "seconds")
        print("Number of iterations: ", len(self.temperature_values))

    def submit_temperature(self, text):
        """Process the temperature entered in the text box"""
        try:
            new_temp = float(text)
            if self.min_temperature <= new_temp <= self.max_temperature:
                corrected_temp = corrected_setpoint(new_temp)
    
                # Guarda valores
                self.original_set_temperature = new_temp
                self.set_temperature = corrected_temp
    
                # Refresca la caja con el valor original introducido
                self.temp_textbox.set_val(f"{new_temp:.1f}")
    
                print(f"User input: {new_temp:.2f}°C → corrected to {corrected_temp:.2f}°C")
    
            else:
                print("Temperature must be between 0°C and 55°C")
        except ValueError:
            print("Insert a valid value")



    def increase_temperature(self, event):
        """Button to increase 0.5°C"""
        new_temp = min(self.max_temperature, self.original_set_temperature + 1.0)
        corrected_temp = corrected_setpoint(new_temp)
        self.original_set_temperature = new_temp
        self.set_temperature = corrected_temp
        self.temp_textbox.set_val(f"{new_temp:.1f}")
        print(f"Setpoint (user): {new_temp:.1f}°C → corrected to {corrected_temp:.2f}°C")

    def decrease_temperature(self, event):
        """Button to decrease 0.5°C"""
        new_temp = max(self.min_temperature, self.original_set_temperature - 1.0)
        corrected_temp = corrected_setpoint(new_temp)
        self.original_set_temperature = new_temp
        self.set_temperature = corrected_temp
        self.temp_textbox.set_val(f"{new_temp:.1f}")
        print(f"Setpoint (user): {new_temp:.1f}°C → corrected to {corrected_temp:.2f}°C")

    def adjust_pid_param(self, param_name, delta):
        """Adjust PID parameters"""
        old_value = getattr(self, param_name)
        new_value = max(0.0, old_value + delta)  # Exclude negative values
        setattr(self, param_name, new_value)
        #print(f"{param_name} updated: {old_value:.3f} → {new_value:.3f}")

        # Update each textbox
        if param_name == 'Kp_cool':
            self.kp_textbox.set_val(f"{new_value:.2f}")
        elif param_name == 'Ki_cool':
            self.ki_textbox.set_val(f"{new_value:.2f}")
        elif param_name == 'Kd_cool':
            self.kd_textbox.set_val(f"{new_value:.2f}")
        
        elif param_name == 'Kp_heat':
            self.kpH_textbox.set_val(f"{new_value:.2f}")
        elif param_name == 'Ki_heat':
            self.kiH_textbox.set_val(f"{new_value:.2f}")
        elif param_name == 'Kd_heat':
            self.kdH_textbox.set_val(f"{new_value:.2f}")

    def update_pid_cooling_from_temp(self, temp):
        """
        Adjust Ki_cool y Kd_cool automatically by using a data fitting
        """
        
        self.Ki_cool = get_ki_cool(temp)
        self.Kd_cool = get_kd_cool(temp)
    
        if hasattr(self, 'ki_textbox'):
            self.ki_textbox.set_val(f"{self.Ki_cool:.2f}")
        if hasattr(self, 'kd_textbox'):
            self.kd_textbox.set_val(f"{self.Kd_cool:.2f}")


    def update_pid_heating_from_temp(self, temp):
        """
        Adjust Ki_heat y Kd_heat automatically by using a data fitting
        """
        self.Ki_heat = get_ki_heat(temp)
        self.Kd_heat = get_kd_heat(temp)
    
        if hasattr(self, 'kiH_textbox'):
            self.kiH_textbox.set_val(f"{self.Ki_heat:.2f}")
        if hasattr(self, 'kdH_textbox'):
            self.kdH_textbox.set_val(f"{self.Kd_heat:.2f}")


    def run(self, num_iterations=4500):
        """Executes the temperature controller"""
        if not self.setup_serial_ports():
            print("Serial ports could not be configured. Aborting.")
            return

        self.start_time = time.time()
        self.last_power_update_time = self.start_time

        try:
            plt.ion()
            fig, ax = plt.subplots(figsize=(11, 8))
            plt.subplots_adjust(bottom=0.3, right=0.8)
            line, = ax.plot([], [], 'b-', label='Temperature')
            hline = ax.axhline(self.original_set_temperature, c='r',
                               ls='--', label='Setpoint')  # 
            
            if self.show_threshold_lines:        
                # Lines for assymetric hysteresis thresholds
                heating_threshold = ax.axhline(self.set_temperature + self.heating_threshold, c='orange', ls=':',
                                               label='Heating threshold')
                cooling_threshold = ax.axhline(self.set_temperature + self.cooling_threshold, c='cyan', ls=':',
                                               label='Cooling threshold')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            ax.tick_params(axis='y', labelsize=14)

            #ax.minorticks_on()  # Add a finer grid
            ax.grid(which='major', linestyle='-', linewidth=0.75)
            ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
            ax.grid(True)
            ax.legend(loc='best', fontsize=14)

            """----------- BUTTONS AND TEXTBOXES FOR THE PLOT--------------"""

            # Up/down buttons for temperature
            ax_decrease = plt.axes([0.05, 0.05, 0.1, 0.075])
            button_decrease = Button(
                ax_decrease, '-', color='lightblue', hovercolor='0.975')
            button_decrease.on_clicked(self.decrease_temperature)

            # Textbox for temperature 
            ax_textbox = plt.axes([0.35, 0.05, 0.2, 0.075])
            self.temp_textbox = TextBox(
                ax_textbox, 'Setpoint: ', initial=f"{self.original_set_temperature:.1f}")
            self.temp_textbox.on_submit(self.submit_temperature)

            # Increasing button
            ax_increase = plt.axes([0.6, 0.05, 0.1, 0.075])
            button_increase = Button(
                ax_increase, '+', color='lightblue', hovercolor='0.975')
            button_increase.on_clicked(self.increase_temperature)

            # Define STOP button
            ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
            button = Button(ax_button, 'STOP', color='red', hovercolor='0.975')
            button.on_clicked(self.stop_simulation)

            # ------------- BUTTONS AND TEXTBOXES FOR COOLING PID -------------

            fig.text(0.9, 0.925, 'COOLING:',
                     fontsize=10, ha='left', va='top')

            # Kp_cool
            ax_kp_val = plt.axes([0.95, 0.85, 0.04, 0.04])
            self.kp_textbox = TextBox(
                ax_kp_val, '$k_p$ = ', initial=f"{self.Kp_cool:.2f}")
            self.kp_textbox.set_active(True)
            
            # Minus button for Kp_cool
            ax_kp_minus = plt.axes([0.83, 0.85, 0.03, 0.04])
            kp_minus_btn = Button(ax_kp_minus, '-', color='lightgray')
            kp_minus_btn.on_clicked(lambda event: self.adjust_pid_param('Kp_cool', -0.01))
            
            # Plus button for Kp_cool
            ax_kp_plus = plt.axes([0.885, 0.85, 0.03, 0.04])
            kp_plus_btn = Button(ax_kp_plus, '+', color='lightgray')
            kp_plus_btn.on_clicked(lambda event: self.adjust_pid_param('Kp_cool', +0.01))


            # Ki_cool
            ax_ki_val = plt.axes([0.95, 0.78, 0.04, 0.04])
            self.ki_textbox = TextBox(
                ax_ki_val, '$k_i$ = ', initial=f"{self.Ki_cool:.2f}")
            self.ki_textbox.set_active(True)
            
            # Minus button for Ki_cool
            ax_ki_minus = plt.axes([0.83, 0.78, 0.03, 0.04])
            ki_minus_btn = Button(ax_ki_minus, '-', color='lightgray')
            ki_minus_btn.on_clicked(lambda event: self.adjust_pid_param('Ki_cool', -0.01))
            
            # Plus button for Ki_cool
            ax_ki_plus = plt.axes([0.885, 0.78, 0.03, 0.04])
            ki_plus_btn = Button(ax_ki_plus, '+', color='lightgray')
            ki_plus_btn.on_clicked(lambda event: self.adjust_pid_param('Ki_cool', +0.01))


            # Kd_cool
            ax_kd_val = plt.axes([0.95, 0.71, 0.04, 0.04])
            self.kd_textbox = TextBox(
                ax_kd_val, '$k_d$ = ', initial=f"{self.Kd_cool:.2f}")
            self.kd_textbox.set_active(True)
            
            # Minus button for Kd_cool
            ax_kd_minus = plt.axes([0.83, 0.71, 0.03, 0.04])
            kd_minus_btn = Button(ax_kd_minus, '-', color='lightgray')
            kd_minus_btn.on_clicked(lambda event: self.adjust_pid_param('Kd_cool', -0.01))
            
            # Plus button for Kd_cool
            ax_kd_plus = plt.axes([0.885, 0.71, 0.03, 0.04])
            kd_plus_btn = Button(ax_kd_plus, '+', color='lightgray')
            kd_plus_btn.on_clicked(lambda event: self.adjust_pid_param('Kd_cool', +0.01))


            # ------------- BUTTONS AND TEXTBOXES FOR HEATING PID -------------

            fig.text(0.9, 0.625, 'HEATING:',
                     fontsize=10, ha='left', va='top')

            # Kp_heat
            ax_kpH_val = plt.axes([0.95, 0.55, 0.04, 0.04])
            self.kpH_textbox = TextBox(
                ax_kpH_val, '$k_p$ = ', initial=f"{self.Kp_heat:.2f}")
            self.kpH_textbox.set_active(True)
            
            # Minus button for Kp_heat
            ax_kpH_minus = plt.axes([0.83, 0.55, 0.03, 0.04])
            kpH_minus_btn = Button(ax_kpH_minus, '-', color='lightgray')
            kpH_minus_btn.on_clicked(lambda event: self.adjust_pid_param('Kp_heat', -0.01))
            
            # Plus button for Kp_heat
            ax_kpH_plus = plt.axes([0.885, 0.55, 0.03, 0.04])
            kpH_plus_btn = Button(ax_kpH_plus, '+', color='lightgray')
            kpH_plus_btn.on_clicked(lambda event: self.adjust_pid_param('Kp_heat', +0.01))


            # Ki_heat
            ax_kiH_val = plt.axes([0.95, 0.48, 0.04, 0.04])
            self.kiH_textbox = TextBox(
                ax_kiH_val, '$k_i$ = ', initial=f"{self.Ki_heat:.2f}")
            self.kiH_textbox.set_active(True)
            
            # Minus button for Ki_heat
            ax_kiH_minus = plt.axes([0.83, 0.48, 0.03, 0.04])
            kiH_minus_btn = Button(ax_kiH_minus, '-', color='lightgray')
            kiH_minus_btn.on_clicked(lambda event: self.adjust_pid_param('Ki_heat', -0.01))
            
            # Plus button for Ki_heat
            ax_kiH_plus = plt.axes([0.885, 0.48, 0.03, 0.04])
            kiH_plus_btn = Button(ax_kiH_plus, '+', color='lightgray')
            kiH_plus_btn.on_clicked(lambda event: self.adjust_pid_param('Ki_heat', +0.01))

            
            # Kd_heat
            ax_kdH_val = plt.axes([0.95, 0.41, 0.04, 0.04])
            self.kdH_textbox = TextBox(
                ax_kdH_val, '$k_d$ = ', initial=f"{self.Kd_heat:.2f}")
            self.kdH_textbox.set_active(True)
            
            # Minus button for Kd_heat
            ax_kdH_minus = plt.axes([0.83, 0.41, 0.03, 0.04])
            kdH_minus_btn = Button(ax_kdH_minus, '-', color='lightgray')
            kdH_minus_btn.on_clicked(lambda event: self.adjust_pid_param('Kd_heat', -0.01))
            
            # Plus button for Kd_heat
            ax_kdH_plus = plt.axes([0.885, 0.41, 0.03, 0.04])
            kdH_plus_btn = Button(ax_kdH_plus, '+', color='lightgray')
            kdH_plus_btn.on_clicked(lambda event: self.adjust_pid_param('Kd_heat', +0.01))


            plt.show()
            



            # ---------- Secondary window for voltage plots -------------------
            if self.enable_extra_plots:

                fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(9, 7))
                fig2.suptitle("Voltage vs Time and Temperature vs Voltage")
                
                line_voltage, = ax2.plot([], [], 'g-', label='Voltage')
                ax2.set_ylabel('Voltage (V)')
                ax2.set_xlabel('Time (s)')
                ax2.grid(True)
                ax2.legend()
                
                line_temp_volt, = ax3.plot([], [], 'm.', label='T vs V')
                ax3.set_ylabel('Temperature (°C)')
                ax3.set_xlabel('Voltage (V)')
                ax3.grid(True)
                ax3.legend()
                
                fig2.tight_layout()
                plt.show()


            while len(self.temperature_values) < num_iterations and not self.stop_flag:
                if self.ser2.in_waiting > 0:
                    try:
                        row = self.ser2.readline().decode().strip()
                        columns = row.split(',')

                        temp = float(columns[0])
                        current_relay_state = columns[1]

                        # Automatically update PID parameters depending on the setpoint
                        self.update_pid_cooling_from_temp(self.set_temperature)
                        self.update_pid_heating_from_temp(self.set_temperature)

                        # Initialize relay state in the first iteration
                        if self.relay_state is None:
                            self.relay_state = current_relay_state
                    except ValueError:
                        continue

                    # Register current time and temperature
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    self.temperature_values.append((elapsed_time, temp))

                    error = temp - self.set_temperature

                    # Determines wheter the relay state should be toggled by asymmetric hysteresis
                    new_relay_state, toggle_needed = self.determine_relay_state(
                        temp, self.set_temperature)

                    # Toggle relay if needed
                    if toggle_needed:
                        self.ser2.write(b'T')
                        print(
                            f"Changing to {'heating' if new_relay_state == 'ON' else 'cooling'} mode")
                        self.relay_state = new_relay_state

                    # Calculate PID output with the dual algorithm
                    output_voltage = self.calculate_pid_output(temp, error)

                    # Send command to the power supply safely
                    self.set_voltage_safely(output_voltage)
                    
                    #
                    self.voltage_values.append((elapsed_time, output_voltage, temp))



                    # Save mode (0: cooling, 1: heating)
                    #mode_value = 1 if self.heating_mode else 0
                    mode_str = "Heating" if self.heating_mode else "Cooling"

                    # Update the title with additional info
                    ax.set_title(f'Dual PID temperature controller - {len(self.temperature_values)} valores\n'
                                 f'Setpoint : {self.set_temperature:.2f}°C\n' 
                                 f'Temperature: {temp:.2f}°C')
                    #ax.set_title(f'Room temperature:  {temp:.2f}°C\n - {len(self.temperature_values)} values\n')
                                 
                    

                    if self.show_threshold_lines:
                        # Update threshold lines when changing setpoint
                        heating_threshold.set_ydata([self.set_temperature + self.heating_threshold,
                                                     self.set_temperature + self.heating_threshold])
                        cooling_threshold.set_ydata([self.set_temperature + self.cooling_threshold,
                                                     self.set_temperature + self.cooling_threshold])

                    # Update graphics with real seconds
                    if len(self.temperature_values) > 5:
                        # Extract time and temperature from data
                        # Real time in seconds
                        x_data = [point[0]
                                  for point in self.temperature_values[5:]]
                        # Temperatures
                        y_data = [point[1]
                                  for point in self.temperature_values[5:]]
                    else:
                        x_data = []
                        y_data = []

                    line.set_xdata(x_data)
                    line.set_ydata(y_data)

                    # Confirm the setpoint horizontal line update
                    hline.set_ydata(
                        [self.original_set_temperature, self.original_set_temperature])

                    ax.relim()
                    ax.autoscale_view(True, True, True)
                    ax.figure.canvas.draw()
                    ax.figure.canvas.flush_events()
                    
                    # Show detailed info
                    temp_rate = self.calculate_temp_rate()
                    print(f"Temperature: {temp:.4f}°C, Mode: {mode_str}, "
                          f"Voltage: {output_voltage:.4f}V, Error: {error:.4f}, "
                          f"Rate: {temp_rate:.4f}°C/sec")
                    
                    # Extract data for the additional plots
                    if self.enable_extra_plots:

                        if len(self.voltage_values) > 5:
                            times_v = [v[0] for v in self.voltage_values[5:]]
                            voltages = [v[1] for v in self.voltage_values[5:]]
                            temps_v = [v[2] for v in self.voltage_values[5:]]
                        else:
                            times_v, voltages, temps_v = [], [], []
                        
                        # Update Voltage vs Time
                        line_voltage.set_xdata(times_v)
                        line_voltage.set_ydata(voltages)
                        ax2.relim()
                        ax2.autoscale_view(True, True, True)
                        
                        # Update Temperature vs Voltage
                        line_temp_volt.set_xdata(voltages)
                        line_temp_volt.set_ydata(temps_v)
                        ax3.relim()
                        ax3.autoscale_view(True, True, True)
                        
                        fig2.canvas.draw()
                        fig2.canvas.flush_events()


        except KeyboardInterrupt:
            print("Connection closed by user.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()



