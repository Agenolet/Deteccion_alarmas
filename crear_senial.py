# -*- coding: utf-8 -*-
"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Función que permite combinar las señales de alarma junto con la señal de 
ruido de UTI para generar señales de testeo.
Para funcionar es necesario contar con los siguientes archivos .wav:
    -Puritan_Bennett_840_x1.wav
    -Newport_HT50_x1.wav
    -Dragger_Carina_x1.wav
    -UTI.wav
Autor: JC
Fecha: Abril 2020

"""

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write

def crear_senial(gNHT50, gPB, gDC, gN, nNHT50, nPB, nDC, sC, sW=False):
    """
    ------------------------
    INPUT:
    --------
    gNHT50: ganancia de la señal correspondiene al respirador Newport (0 a 1) 
    gPB: ganancia de la señal correspondiene al respirador Puritan Bennet (0 a 1) 
    gDC: ganancia de la señal correspondiene al respirador Dragger Carina (0 a 1)
    gN: ganancia de la señal de ruido de fondo (0 a 1)
    nNHT50: cantidad de alarmas a incluir del respirador Newport (0 a 6)
    nPB: cantidad de alarmas a incluir del respirador Puritan Bennet (0 a 6)
    nDC: cantidad de alarmas a incluir del respirador Dragger Carina (0 a 6)
    sC: Coeficiente de solapamiento, 0 no hay, 1 hay, acepta valores intermedios
    sW: True o False, guardar a .wav
    ------------------------
    OUTPUT:
    --------
    data_UTI: señal combinada  
    t: vector de tiempos
    """
    fs, data_PB840 = wavfile.read("./Alarmas/Puritan_Bennett_840_x1.wav")
    fs, data_NHT50 = wavfile.read("./Alarmas/Newport_HT50_x1.wav")
    fs, data_DC = wavfile.read("./Alarmas/Dragger_Carina_x1.wav")
    fs, data_UTI = wavfile.read("./Alarmas/UTI.wav")

    # Definición de parámetro temporales
    ts = 1 / fs                     # tiempo de muestreo
    N = len(data_UTI)                   # número de muestras en el archivo de audio
    data_UTI = data_UTI[1 : int(N/3)]
    N = len(data_UTI)
    t = np.linspace(0, N * ts, N)   # vector de tiempo
    # Converisón de los datos a voltaje
    data_NHT50 = data_NHT50[:, 1]* 3.3 / 2**16             # se extrae un canal de la pista de audio (el audio es estereo)
    data_PB840 = data_PB840[:, 1]* 3.3 / 2**16            
    data_DC = data_DC[:, 1]* 3.3 / 2**16             
    data_UTI = data_UTI[:, 1]* 3.3 / 2**16             

    # Normalización de las señales DISTORSIONA DEMASIADO LA SEÑAL
    #data_NHT50 = data_NHT50/np.max(data_NHT50)
    #data_PB840 = data_PB840/np.max(data_PB840)
    #data_UTI = data_UTI/np.max(data_UTI)
    #data_DC = data_DC/np.max(data_DC)

    # Largo de la señal de ruido
    len_UTI = len(data_UTI)/3
        
    data_UTI =data_UTI*gN
  
    sCo = sC
    if sC <= 0:
        sC = 0.01
    if sC > 1:
        sC = 1
        print("El factor de solapamiento no puede ser mayor a 1")
    if nPB > 6 or nNHT50 > 6 or nDC > 6:
        raise ValueError("La cantidad de alarmas no puede ser mayor a 6")
    if gPB > 1 or gNHT50 > 1 or gDC > 1:
        raise ValueError("La ganancia de alarmas no puede ser mayor a 1")    
    sC = sC**2
    sC =(sC - 0) * (25 - 0.25) / (1 - 0) + 0.25;

    delta_sup_PB840 = 0
    delta_sup_NHT50 = int(len_UTI/(2*sC))
    delta_sup_DC = int(len_UTI/(4*sC))

    for i in range(nPB):
        indice_PB840 = int((len_UTI)/nPB)*i + delta_sup_PB840
        data_UTI[indice_PB840:len(data_PB840)+indice_PB840] = data_PB840*gPB + data_UTI[indice_PB840:len(data_PB840)+ indice_PB840]  

    for i in range(nNHT50):
        indice_NHT50 = int((len_UTI)/nNHT50)*i + delta_sup_NHT50
        data_UTI[indice_NHT50:len(data_NHT50)+indice_NHT50] = data_NHT50*gNHT50 + data_UTI[indice_NHT50:len(data_NHT50)+ indice_NHT50] 

    for i in range(nDC):
        indice_DC = int((len_UTI)/nDC)*i + delta_sup_DC
        data_UTI[indice_DC:len(data_DC)+indice_DC] = data_DC*gDC + data_UTI[indice_DC:len(data_DC)+ indice_DC] 
    
    if sW:
        filename = "Purx" + str(nPB) + "g" + str(gPB) + "_Drax" + str(nDC) + "g" + str(gDC) + "_Newx" + str(nNHT50) + "g" + str(gNHT50) + "_noiseg" +  str(gN) + "_sC" + str(sCo)
        write(filename + ".wav", fs, np.int16(data_UTI/ 3.3 * 2**16))    
    
    return t, data_UTI