# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:30:02 2025

@author: Diego

This code implements a correction for set_temperature based on experimental offset values
"""
# CORRECTED_SETPOINT.py
SHOWGRAPHS = True







import numpy as np
import matplotlib.pyplot as plt

# Observed values of  Tset and the corresponding offset (Treal - Tset)
tset_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
offset_data = np.array([1.5, 1.2, 0.8, 0.4, -0.5, -1.1, -1.5, -1.7])

#offset_data = np.zeros(len(tset_data)) # Case without offset



# Polynomial fit (degree 2) to estimate the offset behavior
offset_fit = np.poly1d(np.polyfit(tset_data, offset_data, 2))

def corrected_setpoint(T_user):
    """
    Returns the corrected setpoint to reduce steady-state error.
    Applies an empirical compensation based on observed offset.
    """
    return T_user - offset_fit(T_user)

# Range of Tset values for visualization
T_range = np.linspace(5, 40, 300)
offset_curve = offset_fit(T_range)
corrected_curve = corrected_setpoint(T_range)


if SHOWGRAPHS == True:
    

    # -------- FIGURE 1: Offset vs Tset --------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(tset_data, offset_data, 'o', label='Measured Offset')
    ax1.plot(T_range, offset_curve, '-', label='Fitted Offset Curve')
    ax1.set_xlabel('User Setpoint $T_{set}$ (°C)')
    ax1.set_ylabel('Offset: $T_{real} - T_{set}$ (°C)')
    ax1.set_title('Measured Offset vs Setpoint', fontsize=18)
    ax1.grid(True)
    ax1.legend(fontsize=18)
    fig1.tight_layout()
    
    # -------- FIGURE 2: Corrected Setpoint vs Original --------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(T_range, T_range, '--', color='gray', label='Ideal: $T_{corrected} = T_{set}$')
    ax2.plot(T_range, corrected_curve, '-', label='Corrected Setpoint')
    ax2.set_xlabel('User Setpoint $T_{set}$ (°C)')
    ax2.set_ylabel('Corrected Setpoint (°C)')
    ax2.set_title('Corrected Setpoint vs User Setpoint', fontsize=18)
    ax2.grid(True)
    ax2.legend(fontsize=18)
    fig2.tight_layout()
    
    plt.show()
    
