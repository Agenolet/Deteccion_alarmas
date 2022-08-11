# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:59:17 2020

@author: Alejandro Genolet
"""
# Librerías
from scipy import fft
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from envolvente import envolvente
from scipy import signal

#Funcion para calcular la fft
#Retorna la señal fft positiva y el vector de frecuencias
def fftmod(senial, N, fs):
    freq = fft.fftfreq(N, d=1/fs)   # se genera el vector de frecuencias
    senial_fft = fft.fft(senial)    # se calcula la transformada rápida de Fourier

    # El espectro es simétrico, nos quedamos solo con el semieje positivo
    f = freq[np.where(freq >= 0)]      
    senial_fft = senial_fft[np.where(freq >= 0)]

    # Se calcula la magnitud del espectro
    senial_fft_mod = np.abs(senial_fft) / N     # Respetando la relación de Parceval
    # Al haberse descartado la mitad del espectro, para conservar la energía 
    # original de la señal, se debe multiplicar la mitad restante por dos (excepto
    # en 0 y fm/2)
    senial_fft_mod[1:len(senial_fft_mod-1)] = 2 * senial_fft_mod[1:len(senial_fft_mod-1)]
    return senial_fft_mod, f 

plt.close('all') # cerrar gráficas anteriores

######################## Lectura del archivo de audio ########################
print('Ingrese el nombre del archivo de audio .wav que desea analizar..')
filename = input()
if ('.wav' not in filename):
    filename += '.wav'

fs, data = wavfile.read('./Alarmas/' + filename)   # frecuencia de muestreo y datos de la señal

# Definición de parámetro temporales
ts = 1 / fs                     # tiempo de muestreo
N = len(data)                   # número de muestras en el archivo de audio
t = np.linspace(0, N * ts, N)   # vector de tiempo
senial = data[:, 1]             # se extrae un canal de la pista de audio (si el audio es estereo)
senial = senial * 3.3 / (2 ** 16 - 1) # se escala la señal a voltios (considerando un CAD de 16bits y Vref 3.3V)

#################### Cálculo de la Transformada de Fourier ###################

senial_fft_mod, f = fftmod(senial, N, fs)

############################### Analisis Temporal #############################
if (filename == 'Dragger_Carina.wav'):
    senial_env = envolvente(senial, 5000)
    maximo = np.max(senial_env)
    
    picos, _ = signal.find_peaks(senial_env, prominence=maximo/4)   # Detección de los picos, retorna el Indice!, valor en eje x
    anchos , altura , pos_inicial, pos_final = signal.peak_widths(senial_env, picos, rel_height=0.5)  # Medición del ancho y posición de los picos
    #La funcion devuelve 4 valores para cada pico:
        #El primero es el ancho en numero de muestras
        #El segundo a que altura lo midió
        #El tercero es el inicio en muestra del pico
        #El cuarto es el fin en muestra del pico
    
    #En el Dragger_Carina un tono tiene 5 "pitidos" en el mismo, se considerará un "tono = los 5 pitidos juntos"
    tono_ancho = pos_final[1] - pos_inicial[0]              #En muestras
    silencio_menor_ancho = pos_inicial[2] - pos_final[1]    #En muestras. Es el ancho entre un tono(5 pitidos), y el siguiente mas corto. Ej el primer silencio entre tono(5 pitidos) y el siguiente tono 
    silencio_mayor_ancho = pos_inicial[4] - pos_final[3]    #En muestras. Es el ancho entre un tono(5 pitidos), y el siguiente mas largo. Ej el 
    
    print(f"El tono de Dragger_Carina dura {tono_ancho*ts:.2f} seg , teniendo en cuenta los 5 tonos.")
    print(f"El silencio mas corto de Dragger_Carina dura {silencio_menor_ancho*ts:.2f} seg.")
    print(f"El silencio mas largo de Dragger_Carina dura {silencio_mayor_ancho*ts:.2f} seg.")
    
if (filename == 'Newport_HT50.wav'):
    senial_env = envolvente(senial, 16000)
    maximo = np.max(senial_env)
    
    picos, _ = signal.find_peaks(senial_env, prominence=maximo/4)   # Detección de los picos, retorna el Indice!, valor en eje x
    anchos , altura , pos_inicial, pos_final = signal.peak_widths(senial_env, picos, rel_height=0.5)  # Medición del ancho y posición de los picos
    #La funcion devuelve 4 valores para cada pico:
        #El primero es el ancho en numero de muestras. El segundo a que altura lo midió. El tercero es el inicio en muestra del pico. El cuarto es el fin en muestra del pico
        
    #El Newport_HT50 tiene 5 pitidos por tono, 3 separados por un pequeño tiempo, y luego de "silencio_menor_ancho" 2 pitidos más.
    #Se repite este tono luego de "silencio_mayor_ancho"
    tono_ancho = pos_final[3] - pos_inicial[0]              #En muestras
    silencio_menor_ancho = pos_inicial[2] - pos_final[1]    #En muestras
    silencio_mayor_ancho = pos_inicial[4] - pos_final[3]    #En muestras
    
    print(f"El tono de NewportHT50 dura {tono_ancho*ts:.2f} seg , teniendo en cuenta los 5 pitidos.")
    print(f"El silencio mas corto, entre los 3 pitidos rapidos y los 2 un poco mas lento, de NewportHT50 dura {silencio_menor_ancho*ts:.2f} seg.")
    print(f"El silencio mas largo de NewportHT50 dura {silencio_mayor_ancho*ts:.2f} seg.")
    
if (filename == 'Puritan_Bennett_840.wav'):
    senial_env = envolvente(senial, 10000)
    maximo = np.max(senial_env)
    
    picos, _ = signal.find_peaks(senial_env, prominence=maximo/4)   # Detección de los picos, retorna el Indice!, valor en eje x
    anchos , altura , pos_inicial, pos_final = signal.peak_widths(senial_env, picos, rel_height=0.5)  # Medición del ancho y posición de los picos
    #La funcion devuelve 4 valores para cada pico:
        #El primero es el ancho en numero de muestras. El segundo a que altura lo midió. El tercero es el inicio en muestra del pico. El cuarto es el fin en muestra del pico
        
    #El Puritan_Bennett_840 tiene 5 pitidos por tono, 3 con frecuencia creciente separados por un pequeño tiempo, y luego de "silencio_menor_ancho" 2 pitidos más iguales.
    #Se repite este tono luego de "silencio_mayor_ancho"
    tono_ancho = pos_final[1] - pos_inicial[0]              #En muestras
    silencio_menor_ancho = pos_inicial[1] - pos_final[0]    #En muestras
    silencio_mayor_ancho = pos_inicial[2] - pos_final[1]    #En muestras
    
    print(f"El tono de Puritan_Bennett_840 dura {tono_ancho*ts:.2f} seg , teniendo en cuenta los 5 pitidos.")
    print(f"El silencio mas corto, entre los 3 pitidos crecientes y los 2 iguales, de Puritan_Bennett_840 dura {silencio_menor_ancho*ts:.2f} seg.")
    print(f"El silencio mas largo de Puritan_Bennett_840 dura {silencio_mayor_ancho*ts:.2f} seg.")
    
    
############################### Graficación ##################################
"""
fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
fig2.suptitle(filename, fontsize=18)

# Se grafica la señal, su envolvente y se señala la duración de los pitidos
ax2.plot(t, senial, label='Señal de audio', zorder=1, color='blue')
ax2.plot(t, senial_env, label='Envolvente', zorder=2, color='green')
ax2.plot(t[picos], senial_env[picos], "X", label='Picos', zorder=3, color='red')
ax2.hlines(altura, pos_inicial * ts, pos_final * ts, label='Duración tono', zorder=4, color="orange")
ax2.set_xlabel('Tiempo [s]', fontsize=15)
ax2.set_ylabel('Tensión [V]', fontsize=15)
ax2.set_xlim([0, ts*N])
ax2.grid()
ax2.legend(fontsize=12)
plt.show()
"""

fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
fig1.suptitle(filename[:len(filename)-4], fontsize=18)  #filename[:len(filename)-4] me elimina la extension .wav

# Se grafica la señal temporal
ax1[0].plot(t, senial, label='Señal de audio', zorder=1, color='blue')
ax1[0].plot(t, senial_env, label='Envolvente', zorder=2, color='green')
ax1[0].plot(t[picos], senial_env[picos], "X", label='Picos', zorder=3, color='red')
ax1[0].hlines(altura, pos_inicial * ts, pos_final * ts, label='Duración tono', zorder=4, color="orange")

ax1[0].set_xlabel('Tiempo [s]', fontsize=15)
ax1[0].set_ylabel('Tensión [V]', fontsize=15)
ax1[0].set_title('Señal temporal', fontsize=15)
ax1[0].set_xlim([0, ts*N])
ax1[0].grid()
ax1[0].legend(fontsize=12)

# Se grafica la magnitud de la respuesta en frecuencia
ax1[1].plot(f, senial_fft_mod)
#ax1[1].plot(f[picos_f], senial_fft_mod[picos_f], "X", label='Picos f', zorder=3, color='red')

ax1[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax1[1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
ax1[1].set_title('Magnitud de la Respuesta en Frecuencia', fontsize=15)
ax1[1].set_xlim([0, fs/2])
ax1[1].grid()
ax1[0].legend(fontsize=12)

plt.show()