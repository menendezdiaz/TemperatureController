# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 13:10:47 2025

@author: Diego
"""

from main import TemperatureController



SETPOINT = 45.0






TemperatureController(set_temperature=SETPOINT).run()


# Fijarse que, para enfriar, el rele debe tener la luz verde encencida, para calentar apagada.
# Si no esta así cambiarlo manualmente antes de ejecutar el codigo

