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
import funciones_fft

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
    
# frecuencia de muestreo y datos de la señal
fs, data = wavfile.read('./Alarmas/' + filename)   

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
    senial_env = envolvente(senial, 100)
    maximo = np.max(senial_env)
    # Detección de los picos, retorna el Indice!, valor en eje x
    picos, _ = signal.find_peaks(senial_env, prominence=maximo/4)   
    # Medición del ancho y posición de los picos
    anchos , altura , pos_inicial, pos_final = signal.peak_widths(senial_env, picos, rel_height=0.5)
        #La funcion devuelve 4 valores para cada pico:
            #El primero es el ancho en numero de muestras
            #El segundo a que altura lo midió
            #El tercero es el inicio en muestra del pico
            #El cuarto es el fin en muestra del pico
    
    #Se considera tono a cada sonido escuchado (es decir, tono = "pitido"). Dragger_Carina tiene 3 pitidos, luego 2 (el ultimo de estos 2 es de menor f, o sea mas grave). Luego se repite estos 5 pitidos.
    #Presenta 4 silencios distintos:
        #·Silencio1c será el que separa a los tonos/pitidos rapidos. Ej, entre los 3 primeros.
        #·Silencio2c será el que separa los 3 tonos/pitidos rapidos, de los 2 que siguen.
        #·Silencio3g será el que separa los 5 tonos/pitidos del siguiente conjunto de 5 tonos/pitidos, del primer conjunto de 5 tonos/pitidos del segundo conjunto, el mas corto de los grandes silencios.
        #·Silencio4g será el que separa los 5 tonos/pitidos del siguiente conjunto de 5 tonos/pitidos, pero el mayor silencio. Ej, entre el segundo conjunto al tercer conjunto. Es el silencio mas grande
    #Todos estos calculos son en Muestras
    tono_ancho = pos_final[0] - pos_inicial[0]        
    silencio1c_ancho = pos_inicial[1] - pos_final[0]    
    silencio2c_ancho = pos_inicial[3] - pos_final[2]   
    silencio3g_ancho = pos_inicial[5] - pos_final[4] 
    silencio4g_ancho = pos_inicial[10] - pos_final[9] 
    
    print('==============')
    print(f"El tono de Dragger_Carina dura {tono_ancho*ts:.2f} seg , teniendo en cuenta 1 tono/pitido.")
    print(f"El silencio entre tonitos de Dragger_Carina dura {silencio1c_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 3 y 2 tonitos de Dragger_Carina dura {silencio2c_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 5 tonitos y los siguientes 5 mas corto de Dragger_Carina dura {silencio3g_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 5 tonitos y los siguientes 5 mas largo de Dragger_Carina dura {silencio4g_ancho*ts:.2f} seg.")
    print('==============')
    
if (filename == 'Newport_HT50.wav'):
    senial_env = envolvente(senial, 50)
    maximo = np.max(senial_env)
    # Detección de los picos, retorna el Indice!, valor en eje x
    picos, _ = signal.find_peaks(senial_env, prominence=maximo/4)   
    # Medición del ancho y posición de los picos
    anchos , altura , pos_inicial, pos_final = signal.peak_widths(senial_env, picos, rel_height=0.5)  
    #La funcion devuelve 4 valores para cada pico:
        #El primero es el ancho en numero de muestras. El segundo a que altura lo midió. El tercero es el inicio en muestra del pico. El cuarto es el fin en muestra del pico
        
    #Se considera tono a cada sonido escuchado (es decir, tono = "pitido"). Newport_HT50 tiene 3 pitidos, luego 2 (los 5 iguales). Luego se repite estos 5 pitidos.
    #Presenta 4 silencios distintos:
        #·Silencio1c será el que separa a los tonos/pitidos rapidos. Ej, entre los 3 primeros.
        #·Silencio2c será el que separa los 3 tonos/pitidos rapidos, de los 2 que siguen.
        #·Silencio3g será el que separa los 5 tonos/pitidos del siguiente conjunto de 5 tonos/pitidos, del primer conjunto de 5 tonos/pitidos del segundo conjunto, el mas corto de los grandes silencios.
        #·Silencio4g será el que separa los 5 tonos/pitidos del siguiente conjunto de 5 tonos/pitidos, pero el mayor silencio. Ej, entre el segundo conjunto al tercer conjunto. Es el silencio mas grande
    
    #Todos estos calculos son en Muestras
    tono_ancho = pos_final[0] - pos_inicial[0]        
    silencio1c_ancho = pos_inicial[1] - pos_final[0]    
    silencio2c_ancho = pos_inicial[3] - pos_final[2]   
    silencio3g_ancho = pos_inicial[5] - pos_final[4] 
    silencio4g_ancho = pos_inicial[10] - pos_final[9] 
    
    print('==============')
    print(f"El tono de Newport_HT50 dura {tono_ancho*ts:.2f} seg , teniendo en cuenta 1 tono/pitido.")
    print(f"El silencio entre tonitos rapidos de Newport_HT50 dura {silencio1c_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 3 y 2 tonitos de Newport_HT50 dura {silencio2c_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 5 tonitos y los siguientes 5 mas corto de Newport_HT50 dura {silencio3g_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 5 tonitos y los siguientes 5 mas largo de Newport_HT50 dura {silencio4g_ancho*ts:.2f} seg.")
    
if (filename == 'Puritan_Bennett_840.wav'):
    senial_env = envolvente(senial, 160)
    maximo = np.max(senial_env)
    # Detección de los picos, retorna el Indice!, valor en eje x
    picos, _ = signal.find_peaks(senial_env, prominence=maximo/4 , height=0.15)   
    # Medición del ancho y posición de los picos
    anchos , altura , pos_inicial, pos_final = signal.peak_widths(senial_env, picos, rel_height=0.76)
    #La funcion devuelve 4 valores para cada pico:
        #El primero es el ancho en numero de muestras. El segundo a que altura lo midió. El tercero es el inicio en muestra del pico. El cuarto es el fin en muestra del pico
        
    #Se considera tono a cada sonido escuchado (es decir, tono = "pitido"). Puritan_Bennett_840 tiene 3 pitidos (f creciente), luego 2 (iguales entre los 2, pero >f que los 3 primeros). Luego se repite estos 5 pitidos.
    #Presenta 4 silencios distintos:
        #·Silencio1c será el que separa a los tonos/pitidos rapidos. Ej, entre los 3 primeros.
        #·Silencio2c será el que separa los 3 tonos/pitidos rapidos, de los 2 que siguen.
        #·Silencio3g será el que separa los 5 tonos/pitidos del siguiente conjunto de 5 tonos/pitidos, del primer conjunto de 5 tonos/pitidos del segundo conjunto, el mas corto de los grandes silencios.
        #·Silencio4g será el que separa los 5 tonos/pitidos del siguiente conjunto de 5 tonos/pitidos, pero el mayor silencio. Ej, entre el segundo conjunto al tercer conjunto. Es el silencio mas grande
    
    #Todos estos calculos son en Muestras
    tono_ancho = pos_final[0] - pos_inicial[0]        
    silencio1c_ancho = pos_inicial[1] - pos_final[0]    
    silencio2c_ancho = pos_inicial[3] - pos_final[2]   
    silencio3g_ancho = pos_inicial[5] - pos_final[4] 
    silencio4g_ancho = pos_inicial[10] - pos_final[9] 
 
    print('==============')
    print(f"El tono de Puritan_Bennett_840 dura {tono_ancho*ts:.2f} seg , teniendo en cuenta 1 tono/pitido.")
    print(f"El silencio entre tonitos rapidos de Puritan_Bennett_840 dura {silencio1c_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 3 y 2 tonitos de Puritan_Bennett_840 dura {silencio2c_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 5 tonitos y los siguientes 5 mas corto de Puritan_Bennett_840 dura {silencio3g_ancho*ts:.2f} seg.")
    print(f"El silencio entre los 5 tonitos y los siguientes 5 mas largo de Puritan_Bennett_840 dura {silencio4g_ancho*ts:.2f} seg.")
    
############################### Analisis Potencia Frecuencial de tonos #############################


if (filename == 'Dragger_Carina.wav'):
    tono1_N = len(senial[int(pos_inicial[0]) : int(pos_final[0])])
    tono2_N = len(senial[int(pos_inicial[4]) : int(pos_final[4])])
    
    tono1_f ,tono1_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[0]) : int(pos_final[0])], fs)
    tono2_f, tono2_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[4]) : int(pos_final[4])], fs)
        
    tono1_armonicos, _ = signal.find_peaks(tono1_fft_mod, distance=100*tono1_N/fs, prominence=np.max(tono1_fft_mod)/50)
    tono2_armonicos, _ = signal.find_peaks(tono2_fft_mod, distance=400*tono2_N/fs, prominence=np.max(tono2_fft_mod)/50) #height=0.008 distance=400*tono1_N/fs
    # Ancho donde encontrar los picos
    tono1_bw =  signal.peak_widths(tono1_fft_mod, tono1_armonicos, rel_height=0.99)[0].astype(int)
    tono2_bw =  signal.peak_widths(tono2_fft_mod, tono2_armonicos, rel_height=0.99)[0].astype(int)
    # Potencia del armonico principal f0
    tono1_pot_f0 = np.sum(tono1_fft_mod[tono1_armonicos[0]-tono1_bw[0] : tono1_armonicos[0]+tono1_bw[0]])
    tono2_pot_f0 = np.sum(tono2_fft_mod[tono2_armonicos[0]-tono2_bw[0] : tono2_armonicos[0]+tono2_bw[0]])
    # Se calcula la sumatoria de la potencia en los armónicos
    # Para tono 1
    tono1_pot_sum_arm = 0
    for i in range(len(tono1_armonicos)-1):
        tono1_pot_sum_arm += np.sum(tono1_fft_mod[tono1_armonicos[i+1]-tono1_bw[i+1] : tono1_armonicos[i+1]+tono1_bw[i+1]])
    # Se calcula la THD1
    tono1_thd = np.sqrt(tono1_pot_sum_arm / tono1_pot_f0) * 100
    # Para Tono 2
    tono2_pot_sum_arm = 0
    for i in range(len(tono2_armonicos)-1):
        tono2_pot_sum_arm += np.sum(tono2_fft_mod[tono2_armonicos[i+1]-tono2_bw[i+1] : tono2_armonicos[i+1]+tono2_bw[i+1]])
    # Se calcula la THD
    tono2_thd = np.sqrt(tono2_pot_sum_arm / tono2_pot_f0) * 100
    
    print(f"Distorsion armonica del tono 1: {tono1_thd:.2f} %")
    print(f"Distorsion armonica del tono 2: {tono2_thd:.2f} %")
    print('==============')
    
    pot_sum_f0 = tono1_pot_f0 + tono2_pot_f0
    pot_armonicos = tono1_pot_sum_arm + tono2_pot_sum_arm
    
if (filename == 'Newport_HT50.wav'):
    tono_N = len(senial[int(pos_inicial[0]) : int(pos_final[0])])
    tono_f ,tono_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[0]) : int(pos_final[0])], fs)
    tono_armonicos, _ = signal.find_peaks(tono_fft_mod, distance=100*tono_N/fs, prominence=np.max(tono_fft_mod)/50)
    # Ancho donde encontrar los picos
    tono_bw = signal.peak_widths(tono_fft_mod, tono_armonicos, rel_height=0.99)[0].astype(int)
    # Potencia del armonico principal f0
    tono_pot_f0 = np.sum(tono_fft_mod[tono_armonicos[0]-tono_bw[0] : tono_armonicos[0]+tono_bw[0]])
    # Se calcula la sumatoria de la potencia en los armónicos
    tono_pot_sum_arm = 0
    for i in range(len(tono_armonicos)-1):
        tono_pot_sum_arm += np.sum(tono_fft_mod[tono_armonicos[i+1]-tono_bw[i+1] : tono_armonicos[i+1]+tono_bw[i+1]])
    # Se calcula la THD1
    tono_thd = np.sqrt(tono_pot_sum_arm / tono_pot_f0) * 100
    
    print(f"Distorsion armonica del tono: {tono_thd:.2f} %")
    print('==============')
    
    pot_sum_f0 = tono_pot_f0
    pot_armonicos = tono_pot_sum_arm
        
if (filename == 'Puritan_Bennett_840.wav'):
    tono1_N = len(senial[int(pos_inicial[0]) : int(pos_final[0])])
    tono2_N = len(senial[int(pos_inicial[1]) : int(pos_final[1])])
    tono3_N = len(senial[int(pos_inicial[2]) : int(pos_final[2])])
    tono4_N = len(senial[int(pos_inicial[3]) : int(pos_final[3])])
    # Se calcula espectro de potencia para cada tono
    tono1_f ,tono1_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[0]) : int(pos_final[0])], fs)
    tono2_f, tono2_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[1]) : int(pos_final[1])], fs)
    tono3_f ,tono3_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[2]) : int(pos_final[2])], fs)
    tono4_f, tono4_fft_mod = funciones_fft.fft_pot(senial[int(pos_inicial[3]) : int(pos_final[3])], fs)    
    # Se calculan picos/armonicos para cada tono
    tono1_armonicos, _ = signal.find_peaks(tono1_fft_mod, distance=100*tono1_N/fs, prominence=np.max(tono1_fft_mod)/50)
    tono2_armonicos, _ = signal.find_peaks(tono2_fft_mod, distance=100*tono2_N/fs, prominence=np.max(tono2_fft_mod)/50)
    tono3_armonicos, _ = signal.find_peaks(tono3_fft_mod, distance=100*tono3_N/fs, prominence=np.max(tono3_fft_mod)/50)
    tono4_armonicos, _ = signal.find_peaks(tono4_fft_mod, distance=100*tono4_N/fs, prominence=np.max(tono4_fft_mod)/50) 
    # Ancho donde encontrar los picos
    tono1_bw =  signal.peak_widths(tono1_fft_mod, tono1_armonicos, rel_height=0.99)[0].astype(int)
    tono2_bw =  signal.peak_widths(tono2_fft_mod, tono2_armonicos, rel_height=0.99)[0].astype(int)
    tono3_bw =  signal.peak_widths(tono3_fft_mod, tono3_armonicos, rel_height=0.99)[0].astype(int)
    tono4_bw =  signal.peak_widths(tono4_fft_mod, tono4_armonicos, rel_height=0.99)[0].astype(int)
    # Potencia del armonico principal f0
    tono1_pot_f0 = np.sum(tono1_fft_mod[tono1_armonicos[0]-tono1_bw[0] : tono1_armonicos[0]+tono1_bw[0]])
    tono2_pot_f0 = np.sum(tono2_fft_mod[tono2_armonicos[0]-tono2_bw[0] : tono2_armonicos[0]+tono2_bw[0]])
    tono3_pot_f0 = np.sum(tono3_fft_mod[tono3_armonicos[0]-tono3_bw[0] : tono3_armonicos[0]+tono3_bw[0]])
    tono4_pot_f0 = np.sum(tono4_fft_mod[tono4_armonicos[0]-tono4_bw[0] : tono4_armonicos[0]+tono4_bw[0]])
    # Se calcula la sumatoria de la potencia en los armónicos
    # Para tono 1
    tono1_pot_sum_arm = 0
    for i in range(len(tono1_armonicos)-1):
        tono1_pot_sum_arm += np.sum(tono1_fft_mod[tono1_armonicos[i+1]-tono1_bw[i+1] : tono1_armonicos[i+1]+tono1_bw[i+1]])
    # Se calcula la THD1
    tono1_thd = np.sqrt(tono1_pot_sum_arm / tono1_pot_f0) * 100
    # Para Tono 2
    tono2_pot_sum_arm = 0
    for i in range(len(tono2_armonicos)-1):
        tono2_pot_sum_arm += np.sum(tono2_fft_mod[tono2_armonicos[i+1]-tono2_bw[i+1] : tono2_armonicos[i+1]+tono2_bw[i+1]])
    # Se calcula la THD2
    tono2_thd = np.sqrt(tono2_pot_sum_arm / tono2_pot_f0) * 100
    # Para tono 3
    tono3_pot_sum_arm = 0
    for i in range(len(tono3_armonicos)-1):
        tono3_pot_sum_arm += np.sum(tono3_fft_mod[tono3_armonicos[i+1]-tono3_bw[i+1] : tono3_armonicos[i+1]+tono3_bw[i+1]])
    # Se calcula la THD3
    tono3_thd = np.sqrt(tono3_pot_sum_arm / tono3_pot_f0) * 100
    # Para Tono 4
    tono4_pot_sum_arm = 0
    for i in range(len(tono4_armonicos)-1):
        tono4_pot_sum_arm += np.sum(tono4_fft_mod[tono4_armonicos[i+1]-tono4_bw[i+1] : tono4_armonicos[i+1]+tono4_bw[i+1]])
    # Se calcula la THD4
    tono4_thd = np.sqrt(tono4_pot_sum_arm / tono4_pot_f0) * 100
    
    print(f"Distorsion armonica del tono 1: {tono1_thd:.2f} %")
    print(f"Distorsion armonica del tono 2: {tono2_thd:.2f} %")
    print(f"Distorsion armonica del tono 3: {tono3_thd:.2f} %")
    print(f"Distorsion armonica del tono 4: {tono4_thd:.2f} %")
    print('==============')
    
    pot_sum_f0 = tono1_pot_f0 + tono2_pot_f0 + tono3_pot_f0 + tono4_pot_f0
    pot_armonicos = tono1_pot_sum_arm + tono2_pot_sum_arm + tono3_pot_sum_arm + tono4_pot_sum_arm
  
   
# Calculo para Relacion Señal-Ruido


############################### Graficación ##################################
fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
fig1.suptitle(filename[:len(filename)-4], fontsize=18)  #filename[:len(filename)-4] me elimina la extension .wav

# Se grafica la señal temporal
ax1[0].plot(t, senial, label='Señal de audio', zorder=1, color='blue')
ax1[0].plot(t, senial_env, label='Envolvente', zorder=2, color='green')
ax1[0].plot(t[picos], senial_env[picos], "X", label='Picos', zorder=3, color='red')
ax1[0].hlines(altura, pos_inicial*ts, pos_final*ts, label='Duración tono', zorder=4, color="orange")
ax1[0].set_xlabel('Tiempo [s]', fontsize=15)
ax1[0].set_ylabel('Tensión [V]', fontsize=15)
ax1[0].set_title('Señal temporal', fontsize=15)
#ax1[0].set_xlim([0, ts*N])
ax1[0].set_xlim([0, ts*(N-4*N/5)]) #Para graficar solo una "melodia"
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
ax1[1].legend(fontsize=12)

# Graficacion de tonos
if (filename == 'Dragger_Carina.wav'):
    fig2, ax2 = plt.subplots(2, 2, figsize=(20, 10))
    fig1.suptitle('Tonos de ' + filename[:len(filename)-4], fontsize=18)  #filename[:len(filename)-4] me elimina la extension .wav

    ax2[0,0].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0,0].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0,0].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0,0].set_title('Tono 1', fontsize=15)
    ax2[0,0].set_xlim([ts*int(pos_inicial[0]), ts*int(pos_final[0])])
    ax2[0,0].grid()
    ax2[0,0].legend(fontsize=12)
    
    ax2[0,1].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0,1].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0,1].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0,1].set_title('Tono 2', fontsize=15)
    ax2[0,1].set_xlim([ts*int(pos_inicial[4]) , ts*int(pos_final[4])])
    ax2[0,1].grid()
    ax2[0,1].legend(fontsize=12)
    
    # Se grafica la espectros de potencia en frecuencia para cada tono
    ax2[1,0].plot(tono1_f, tono1_fft_mod)
    ax2[1,0].plot(tono1_f[tono1_armonicos], tono1_fft_mod[tono1_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1,0].plot(tono1_f[tono1_armonicos[0]-tono1_bw[0] : tono1_armonicos[0]+tono1_bw[0]] ,tono1_fft_mod[tono1_armonicos[0]-tono1_bw[0] : tono1_armonicos[0]+tono1_bw[0]], color='orange')
    ax2[1,0].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1,0].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1,0].set_title('Espectro de Potencia Tono 1', fontsize=15)
    ax2[1,0].set_xlim([0, 10000])
    ax2[1,0].grid()
    ax2[1,0].legend(fontsize=12)
    
    ax2[1,1].plot(tono2_f, tono2_fft_mod)
    ax2[1,1].plot(tono2_f[tono2_armonicos], tono2_fft_mod[tono2_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1,1].plot(tono2_f[tono2_armonicos[0]-tono2_bw[0] : tono2_armonicos[0]+tono2_bw[0]] ,tono2_fft_mod[tono2_armonicos[0]-tono2_bw[0] : tono2_armonicos[0]+tono2_bw[0]], color='orange')
    ax2[1,1].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1,1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1,1].set_title('Espectro de Potencia Tono 2', fontsize=15)
    ax2[1,1].set_xlim([0, 10000])
    ax2[1,1].grid()
    ax2[1,1].legend(fontsize=12)

if (filename == 'Newport_HT50.wav'):
    fig2, ax2 = plt.subplots(2, 1, figsize=(20, 10))
    fig1.suptitle('Tonos de ' + filename[:len(filename)-4], fontsize=18)  #filename[:len(filename)-4] me elimina la extension .wav

    ax2[0].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0].set_title('Tono', fontsize=15)
    ax2[0].set_xlim([ts*int(pos_inicial[0]), ts*int(pos_final[0])])
    ax2[0].grid()
    ax2[0].legend(fontsize=12)
    
    # Se grafica la espectros de potencia en frecuencia del tono
    ax2[1].plot(tono_f, tono_fft_mod)
    ax2[1].plot(tono_f[tono_armonicos], tono_fft_mod[tono_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1].plot(tono_f[tono_armonicos[0]-tono_bw[0] : tono_armonicos[0]+tono_bw[0]] ,tono_fft_mod[tono_armonicos[0]-tono_bw[0] : tono_armonicos[0]+tono_bw[0]], color='orange')
    ax2[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1].set_title('Espectro de Potencia de Tono', fontsize=15)
    ax2[1].set_xlim([0, 10000])
    ax2[1].grid()
    ax2[1].legend(fontsize=12)
 
if (filename == 'Puritan_Bennett_840.wav'):
    fig2, ax2 = plt.subplots(2, 4, figsize=(20, 10))
    fig1.suptitle('Tonos de ' + filename[:len(filename)-4], fontsize=18)  #filename[:len(filename)-4] me elimina la extension .wav

    ax2[0,0].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0,0].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0,0].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0,0].set_title('Tono 1', fontsize=15)
    ax2[0,0].set_xlim([ts*int(pos_inicial[0]), ts*int(pos_final[0])])
    ax2[0,0].grid()
    ax2[0,0].legend(fontsize=12)
    
    ax2[0,1].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0,1].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0,1].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0,1].set_title('Tono 2', fontsize=15)
    ax2[0,1].set_xlim([ts*int(pos_inicial[1]) , ts*int(pos_final[1])])
    ax2[0,1].grid()
    ax2[0,1].legend(fontsize=12)
    
    ax2[0,2].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0,2].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0,2].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0,2].set_title('Tono 3', fontsize=15)
    ax2[0,2].set_xlim([ts*int(pos_inicial[2]), ts*int(pos_final[2])])
    ax2[0,2].grid()
    ax2[0,2].legend(fontsize=12)
    
    ax2[0,3].plot(t, senial, label='Señal audio', zorder=1, color='blue')
    ax2[0,3].set_xlabel('Tiempo [s]', fontsize=15)
    ax2[0,3].set_ylabel('Tensión [V]', fontsize=15)
    ax2[0,3].set_title('Tono 4', fontsize=15)
    ax2[0,3].set_xlim([ts*int(pos_inicial[3]), ts*int(pos_final[3])])
    ax2[0,3].grid()
    ax2[0,3].legend(fontsize=12)
    
    # Se grafica la espectros de potencia en frecuencia para cada tono
    ax2[1,0].plot(tono1_f, tono1_fft_mod)
    ax2[1,0].plot(tono1_f[tono1_armonicos], tono1_fft_mod[tono1_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1,0].plot(tono1_f[tono1_armonicos[0]-tono1_bw[0] : tono1_armonicos[0]+tono1_bw[0]] ,tono1_fft_mod[tono1_armonicos[0]-tono1_bw[0] : tono1_armonicos[0]+tono1_bw[0]], color='orange')
    ax2[1,0].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1,0].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1,0].set_title('Espectro de Potencia Tono 1', fontsize=15)
    ax2[1,0].set_xlim([0, 10000])
    ax2[1,0].grid()
    ax2[1,0].legend(fontsize=12)
    
    ax2[1,1].plot(tono2_f, tono2_fft_mod)
    ax2[1,1].plot(tono2_f[tono2_armonicos], tono2_fft_mod[tono2_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1,1].plot(tono2_f[tono2_armonicos[0]-tono2_bw[0] : tono2_armonicos[0]+tono2_bw[0]] ,tono2_fft_mod[tono2_armonicos[0]-tono2_bw[0] : tono2_armonicos[0]+tono2_bw[0]], color='orange')
    ax2[1,1].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1,1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1,1].set_title('Espectro de Potencia Tono 2', fontsize=15)
    ax2[1,1].set_xlim([0, 10000])
    ax2[1,1].grid()
    ax2[1,1].legend(fontsize=12)
    
    ax2[1,2].plot(tono3_f, tono3_fft_mod)
    ax2[1,2].plot(tono3_f[tono3_armonicos], tono3_fft_mod[tono3_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1,2].plot(tono3_f[tono3_armonicos[0]-tono3_bw[0] : tono3_armonicos[0]+tono3_bw[0]] ,tono3_fft_mod[tono3_armonicos[0]-tono3_bw[0] : tono3_armonicos[0]+tono3_bw[0]], color='orange')
    ax2[1,2].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1,2].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1,2].set_title('Espectro de Potencia Tono 3', fontsize=15)
    ax2[1,2].set_xlim([0, 10000])
    ax2[1,2].grid()
    ax2[1,2].legend(fontsize=12)
    
    ax2[1,3].plot(tono4_f, tono4_fft_mod)
    ax2[1,3].plot(tono4_f[tono4_armonicos], tono4_fft_mod[tono4_armonicos], "X", label='Picos f', zorder=3, color='red')
    ax2[1,3].plot(tono4_f[tono4_armonicos[0]-tono4_bw[0] : tono4_armonicos[0]+tono4_bw[0]] ,tono4_fft_mod[tono4_armonicos[0]-tono4_bw[0] : tono4_armonicos[0]+tono4_bw[0]], color='orange')
    ax2[1,3].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax2[1,3].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax2[1,3].set_title('Espectro de Potencia Tono 4', fontsize=15)
    ax2[1,3].set_xlim([0, 10000])
    ax2[1,3].grid()
    ax2[1,3].legend(fontsize=12)
  

plt.show()






