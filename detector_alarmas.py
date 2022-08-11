# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Autor: Albano Peñalva
Fecha: Mayo 2020

"""

# Librerías
import numpy as np
from scipy import signal

class DetectorAlarmas:
    """
    Clase  que contiene funciones para la detección de eventos de alarmas de
    respiradores.
    Los algoritmos para la detección de las alarmas están basados en la 
    utilización de Filtros "Matched" sobre la envolvente de la señal de 
    entrada.
    El Filtrado "Matched" en cuestión consiste en realizar una correlación 
    entre la envolvente de la señal de entrada y una "plantilla" con la "forma"
    de la alarma a detectar.
    Cuando la correlación entre ambas señales es máxima, se determina que se 
    ha detectado una alarma.
    """

    __F_SUB = 100 # Frecuencia de submuestreo de la señal luego de la detección de envolvente.
 
    __umbral = None # Umbral para detectar ocurrencias de alarma a la salida del filtro matched
    __duracion = None # Duración total de la secuencia de la alarma
    __fs = None # Frecuencia de muestreo de la señal de entrada
    __template = np.array([]) # Plantilla con la "forma" de la alarma a detectar
    # Filtro pasabajos para detección de envolvente
    __num_env  = np.array([]) 
    __den_env  = np.array([])
    
    def __init__(self, RoT, fs, umbral=None):
        """
        ------------------------
        INPUT:
        --------
        RoT: {String, array}
            Si es un string con las opciones 'Newport', 'Dragger' o 'Puritan'
            utiliza una plantilla predeterminada para la alarma seleccionada.
            Si es un array utiliza los valores de tiempos para cada pitido y 
            silencio en RoT (en milisegundos) para generar una nueva plantilla:
             ___     ___     ___        ___     ___
            |   |   |   |   |   |      |   |   |   |
            |   |___|   |___|   |______|   |___|   |
             TH1 TL1 TH2 TL2 TH3  TL3   TH4 TL4 TH5    
         
            Donde TH1 = RoT[0], TL1 = RoT[1], TH2 = RoT[2], ...
        fs: int
            Frecuencia de muestreo de la señal.
        umbral: float (opcional)
            Valor de referencia usado en el algoritmo para decidir si se ha 
            detectado una alarma. Si su valor es None, se utilizan valores 
            pre-establecidos.
        ------------------------
        OUTPUT:
        --------
        None
         
        """
                
        self.__fs = fs
        
        if RoT == 'Newport':
             TH1 = 175
             TL1 = 200
             TH2 = 200
             TL2 = 200
             TH3 = 200
             TL3 = 400
             TH4 = 200
             TL4 = 200
             TH5 = 200
             self.__umbral = 15
        
        elif RoT == 'Dragger':
             TH1 = 120
             TL1 = 50
             TH2 = 120
             TL2 = 50
             TH3 = 120
             TL3 = 250
             TH4 = 120
             TL4 = 50
             TH5 = 120
             self.__umbral = 3
             
        elif RoT == 'Puritan':
             TH1 = 175
             TL1 = 75
             TH2 = 100
             TL2 = 75
             TH3 = 175
             TL3 = 135
             TH4 = 175
             TL4 = 75
             TH5 = 175
             self.__umbral = 2

        else:
             TH1 = RoT[0]
             TL1 = RoT[1]
             TH2 = RoT[2]
             TL2 = RoT[3]
             TH3 = RoT[4]
             TL3 = RoT[5]
             TH4 = RoT[6]
             TL4 = RoT[7]
             TH5 = RoT[8]
             self.__umbral = 5
             
        if umbral != None:
            self.__umbral = umbral
             
        # Duración total de la alarma
        self.__duracion = TH1 + TL1 + TH2 + TL2 + TH3 + TL3 + TH4 + TL4 + TH5 
        
        # Se construye la plantilla
        n1 = int(TH1 * self.__F_SUB / 1000)
        n2 = int((TH1 + TL1) * self.__F_SUB / 1000)
        n3 = int((TH1 + TL1 + TH2) * self.__F_SUB / 1000)
        n4 = int((TH1 + TL1 + TH2 + TL2) * self.__F_SUB / 1000)
        n5 = int((TH1 + TL1 + TH2 + TL2 + TH3) * self.__F_SUB / 1000)
        n6 = int((TH1 + TL1 + TH2 + TL2 + TH3 + TL3) * self.__F_SUB / 1000)
        n7 = int((TH1 + TL1 + TH2 + TL2 + TH3 + TL3 + TH4) * self.__F_SUB / 1000)
        n8 = int((TH1 + TL1 + TH2 + TL2 + TH3 + TL3 + TH4 + TL4) * self.__F_SUB / 1000)
        self.__template = np.zeros(int(self.__duracion*self.__F_SUB/1000))
        self.__template[0 : n1] = 1
        self.__template[n2 : n3] = 1
        self.__template[n4 : n5] = 1
        self.__template[n6 : n7] = 1
        self.__template[n8 : ] = 1
        
        # Se invierte la plantilla para ser usada como coeficientes del Filtro Matched
        self.__template = self.__template[::-1]
        
        # Pasabajos de orden 4 y frecuencia de corte en 10Hz para detección de envolvente
        self.__num_env, self.__den_env = signal.butter(4, 2*10/fs, 'low') 
        

    def detectar_alarma(self, senial):
        """
        ------------------------
        INPUT:
        --------
        senial: array 
            Señal a partir de la cual se busca detectar la ocurrencia de alarmas.
        ------------------------
        OUTPUT:
        --------
        ocurrencias: array
            Valores de tiempo (en seg) donde se detectan las ocurrencias de la 
            alarma.
        """
    
        N = len(senial)
        
        # Se realiza la detección de envolvente a partir de aplicar un 
        # pasabajos a la señal rectificada
        senial_env = signal.lfilter(self.__num_env, self.__den_env, abs(senial))
        
        # Se submuestrea la señal con el objetivo de disminuir el costo 
        # computacional del filtrado posterior. Esto es posible ya que la 
        # detección de envolvente eliminó las componetes frecunciales superiores.
        senial_submuest = signal.resample(senial_env, int(N*self.__F_SUB/self.__fs))
        
        # Se aplica el filtrado "Matched" 
        senial_filtrada = signal.lfilter(self.__template, 1, senial_submuest)
        
        # Se analiza la ubicación de los picos en la señal resultante para 
        # determinar las ocurrencias de la alarma.
        ocurrencias, _ = signal.find_peaks(senial_filtrada, 
                                           prominence=self.__umbral,
                                           distance=self.__duracion*self.__F_SUB/1000)
        
        # Se traducen lso valores a segundos
        ocurrencias = ocurrencias / self.__F_SUB
        
        return ocurrencias