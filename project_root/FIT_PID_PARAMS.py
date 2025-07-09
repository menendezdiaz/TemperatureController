# -*- coding: utf-8 -*-
"""
Created on July 1, 2025

@author: Diego

This script performs a second-degree polynomial fit for Ki and Kd parameters
using 10 subsampled measurements per temperature for both cooling and heating modes.
"""



SHOWGRAPHS = False




import numpy as np
import matplotlib.pyplot as plt

# Temperature arrays
temps_cool = np.array([3, 6, 9, 12, 15, 18])
temps_heat = np.array([24, 28, 32, 36, 40, 44])


# Experimental values which made the controller work well enough:

ki_cool_data = np.array([
    [0.40, 0.37, 0.35, 0.35, 0.40, 0.37],
    [0.28, 0.30, 0.30, 0.26, 0.26, 0.25],
    [0.14, 0.16, 0.16, 0.14, 0.16, 0.12],
    [0.10, 0.08, 0.08, 0.10, 0.12, 0.10],
    [0.04, 0.02, 0.00, 0.04, 0.02, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
])

kd_cool_data = np.array([
    [0.04, 0.02, 0.04, 0.04, 0.04, 0.02],
    [0.04, 0.03, 0.04, 0.04, 0.02, 0.03],
    [0.02, 0.02, 0.03, 0.01, 0.02, 0.03],
    [0.01, 0.02, 0.00, 0.00, 0.01, 0.01],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
])

ki_heat_data = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.02, 0.01, 0.02, 0.02, 0.00, 0.01],
    [0.05, 0.05, 0.05, 0.03, 0.05, 0.08],
    [0.12, 0.10, 0.12, 0.10, 0.12, 0.12],
    [0.20, 0.25, 0.22, 0.20, 0.22, 0.20],
    [0.30, 0.32, 0.30, 0.30, 0.28, 0.32],
])

kd_heat_data = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.01, 0.00, 0.00, 0.01, 0.00, 0.00],
    [0.02, 0.00, 0.02, 0.02, 0.02, 0.00],
    [0.02, 0.02, 0.02, 0.03, 0.01, 0.02],
])



# Compute means
ki_cool_avg = np.mean(ki_cool_data, axis=1)
kd_cool_avg = np.mean(kd_cool_data, axis=1)
ki_heat_avg = np.mean(ki_heat_data, axis=1)
kd_heat_avg = np.mean(kd_heat_data, axis=1)

# Compute standard deviations for errorbars
ki_cool_err = np.std(ki_cool_data, axis=1, ddof=2)
kd_cool_err = np.std(kd_cool_data, axis=1, ddof=2)
ki_heat_err = np.std(ki_heat_data, axis=1, ddof=2)
kd_heat_err = np.std(kd_heat_data, axis=1, ddof=2)


# Evaluation functions with embedded polynomial fits and suppression between 18–25 °C

def get_ki_cool(temp):
    coeffs = np.polyfit(temps_cool, ki_cool_avg, deg=2)
    if temp >= 18:
        return 0.0
    return max(0.0, np.polyval(coeffs, temp)) 

def get_kd_cool(temp):
    coeffs = np.polyfit(temps_cool[:-1], kd_cool_avg[:-1], deg=2)
    if temp >= 15:
        return 0.0
    return max(0.0, np.polyval(coeffs, temp))

def get_ki_heat(temp):
    coeffs = np.polyfit(temps_heat, ki_heat_avg, deg=2)
    if temp <= 25:
        return 0.0
    return max(0.0, np.polyval(coeffs, temp))

def get_kd_heat(temp):
    coeffs = np.polyfit(temps_heat[2:], kd_heat_avg[2:], deg=2)
    if temp <= 30:
        return 0.0
    return max(0.0, np.polyval(coeffs, temp))




if SHOWGRAPHS == True:

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t_cool = np.linspace(min(temps_cool), max(temps_cool), 200)
    t_heat = np.linspace(min(temps_heat), max(temps_heat), 200)

    # Ki cooling
    ax = axes[0, 0]
    ax.errorbar(temps_cool, ki_cool_avg, yerr=ki_cool_err, fmt='ko', alpha=0.7, label='Data $K_d$', capsize=3)
    ax.plot(temps_cool, ki_cool_avg, 'ko', label='Mean values')
    ax.plot(t_cool, [get_ki_cool(t) for t in t_cool], 'b-', label=r'Effective $K_i^{cool}(\tau)$')
    ax.set_title(r"$K_i^{cool}$ vs $\tau$ fit", fontsize=18)
    ax.set_xlabel(r"Temperature setpoint $\tau$(°C)")
    ax.tick_params(axis='y', labelsize=14)

    ax.set_ylabel("$K_i$")
    ax.grid(True)
    ax.legend(fontsize=18)


    # Kd cooling
    ax = axes[1, 0]
    ax.errorbar(temps_cool, kd_cool_avg, yerr=kd_cool_err, fmt='ko', alpha=0.7, label='Data $K_d$', capsize=3)
    ax.plot(temps_cool, kd_cool_avg, 'ko', label='Mean values')
    ax.plot(t_cool, [get_kd_cool(t) for t in t_cool], 'b-', label=r'Effective $K_d^{cool}(\tau)$')
    ax.set_title(r"$K_d^{cool}$ vs $\tau$ fit", fontsize=18)
    ax.set_xlabel(r"Temperature setpoint $\tau$(°C)")
    ax.set_ylabel("$K_d$")
    ax.grid(True)
    ax.legend(fontsize=18)

    # Ki heating
    ax = axes[0, 1]
    ax.errorbar(temps_heat, ki_heat_avg, yerr=ki_heat_err, fmt='ko', alpha=0.7, label='Data $K_i$', capsize=3)
    ax.plot(temps_heat, ki_heat_avg, 'ko', label='Mean values')
    ax.plot(t_heat, [get_ki_heat(t) for t in t_heat], 'r-', label=r'Effective $K_i^{heat}(\tau)$')
    ax.set_title(r"$K_i^{heat}$ vs $\tau$ fit", fontsize=18)
    ax.set_xlabel(r"Temperature setpoint $\tau$(°C)")
    ax.set_ylabel("$K_i$")
    ax.grid(True)
    ax.legend(fontsize=18)

    # Kd heating
    ax = axes[1, 1]
    ax.errorbar(temps_heat, kd_heat_avg, yerr=kd_heat_err, fmt='ko', alpha=0.7, label='Data $K_d$', capsize=3)

    ax.plot(temps_heat, kd_heat_avg, 'ko', label='Mean values')
    ax.plot(t_heat, [get_kd_heat(t) for t in t_heat], 'r-', label=r'Effective $K_d^{heat}(\tau)$')
    ax.set_title(r"$K_d^{heat}$ vs $\tau$ fit", fontsize=18)
    ax.set_xlabel(r"Temperature setpoint $\tau$(°C)")
    ax.set_ylabel("$K_d$")
    ax.grid(True)
    ax.legend(fontsize=18)

    plt.tight_layout()
    plt.show()
