# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Función que devuelve la envolvente de una señal x, obtenida a partir del 
cálculo del valor RMS de dicha señal sobre una ventana móvi. El ancho de la
ventana utilizada es determinado por el parámetro vent. 

Autor: Albano Peñalva
Fecha: Marzo 2020

"""

# Librerías
from scipy import signal
import numpy as np

def envolvente(x, vent):
    """
    ------------------------
    INPUT:
    --------
    x: array de una dimensión conteniendo la señal cuya envolvente se busca 
    extraer
    vent: numero de muestras de la ventana móvil a aplicar
    ------------------------
    OUTPUT:
    --------
    env: array de una dimensión conteniendo la envolvente superior de la señal
    """
    
    # RMS = √(1/N * Σx²(i))
    
    # Se define un arreglo de N elementos de valor 1/N
    divisor = [1/vent for i in range(vent)] 
    # Se eleva cada elemento de la señal al cuadrado
    x2 = np.power(x, 2) 
    # Se realiza el promedio movil de las señal elevada al cuadrado 
    prom_mov = signal.lfilter(divisor, 1, x2)
    # Se aplica la raiz cuadrada
    env = np.sqrt(prom_mov)
    # Se escala al valor máximo de la señal
    env = np.sqrt(prom_mov) * np.max(x) / np.max(env)

    return env
