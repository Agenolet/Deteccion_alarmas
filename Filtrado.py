# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:05:57 2020

@author: Alejandro Genolet
"""
# Librerías
from scipy import fft
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
# from envolvente import envolvente
from scipy import signal
# import funciones_fft

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

############################### Etapa 2: Filtros #############################  
#Se trabaja para distingir la Dragger_Carina del resto de alarmas, y de un ambiente ruidoso en general

#Cargo el filtro para tono 1 - f0
filtro_iir_t1_f0 = np.load('./Filtros/' + 'Dragger_Carina_t1_f0.npz', allow_pickle=True)
Num_iir_t1_f0, Den_iir_t1_f0 = filtro_iir_t1_f0['ba'] 
# #Cargo el filtro para tono 1 - f1
filtro_iir_t1_f1 = np.load('./Filtros/' + 'Dragger_Carina_t1_f1.npz', allow_pickle=True)
Num_iir_t1_f1, Den_iir_t1_f1 = filtro_iir_t1_f1['ba'] 
# #Cargo el filtro para tono 1 - f2
filtro_iir_t1_f2 = np.load('./Filtros/' + 'Dragger_Carina_t1_f2.npz', allow_pickle=True)
Num_iir_t1_f2, Den_iir_t1_f2 = filtro_iir_t1_f2['ba'] 
# #Cargo el filtro para tono 2 - f2
filtro_iir_t2_f2 = np.load('./Filtros/' + 'Dragger_Carina_t2_f2.npz', allow_pickle=True)
Num_iir_t2_f2, Den_iir_t2_f2 = filtro_iir_t2_f2['ba'] 

#Genero una señal como la suma de varias señales de alarmas. El if es para poder desplejar el codigo y acortarlo
if 1==0:
    # filename_Dra = 'Dragger_Carina.wav'          
    # filename_New = 'Newport_HT50.wav' 
    # filename_Pur = 'Puritan_Bennett_840.wav' 
    # filename_UTI = 'UTI.wav'
    filename_n = ('Dragger_Carina.wav','Newport_HT50.wav','Puritan_Bennett_840.wav','UTI.wav')
    fs_n=[0,0,0,0]
    data_n=[[],[],[],[]]
    ts_n=[0,0,0,0]
    N_n=[0,0,0,0]
    t_n=[[],[],[],[]]
    senial_n=[[],[],[],[]]
    
    for i in range(4):
        fs_n[i], data_n[i] = wavfile.read('./Alarmas/' + filename_n[i])
        ts_n[i] = 1 / fs_n[i]                   
        N_n[i] = len(data_n[i])               
        t_n[i] = np.linspace(0, N_n[i] * ts_n[i], N_n[i])  
        senial_n[i] = data_n[i][:, 1] 
        senial_n[i] = senial_n[i] * 3.3 / (2 ** 16 - 1)
        
    #Recorto las señales mas largas, para que todas tengan la misma longitud  
    largo_menor = N_n[0]
    for i in range(3):
        if (largo_menor > N_n[i+1]):
            largo_menor = N_n[i+1]
            
    for i in range(4):
        senial_n[i] = senial_n[i][0:largo_menor]
    
    #Sumo las señales de alarmas y uti
    senial_sumada = []
    for i in range(largo_menor):
    	senial_sumada.append(senial_n[0][i]+senial_n[1][i]+senial_n[2][i]+senial_n[3][i])

    fs_sumada = fs_n[0]
    N_senial_sumada = largo_menor
    
    #Genero un .wav de la senial SUMADA
    senial_sumada_wav=senial_sumada.astype('float32')
    wavfile.write('Senial_Sumada.wav',fs_sumada,senial_sumada)

#Cargo la señal que quiero filtrar

# filename_a_filtrar = 'UTI.wav'      
# filename_a_filtrar = 'Dragger_Carina.wav'  
# filename_a_filtrar = 'Newport_HT50.wav'  
# filename_a_filtrar = 'Puritan_Bennett_840.wav'         
# filename_a_filtrar = 'Dragger_Carina_UTI.wav' 
# filename_a_filtrar = 'Senial_Sumada.wav' 
         
fs_a_filtrar, data_a_filtrar = wavfile.read('./Alarmas/' + filename_a_filtrar)   
ts_a_filtrar = 1 / fs_a_filtrar                   
N_a_filtrar = len(data_a_filtrar)               
t_a_filtrar = np.linspace(0, N_a_filtrar * ts_a_filtrar, N_a_filtrar)  
senial_a_filtrar = data_a_filtrar[:, 1] 
senial_a_filtrar = senial_a_filtrar * 3.3 / (2 ** 16 - 1)

# senial_a_filtrar = senial_sumada
# fs_a_filtrar = fs_n[0]
# ts_a_filtrar = 1 / fs_a_filtrar
# N_a_filtrar = N_senial_sumada
# t_a_filtrar = np.linspace(0, N_a_filtrar * ts_a_filtrar, N_a_filtrar) 

#===============Aplicacion del filtro===============
#Se aplica el filtrado para f0 sobre la señal LIMPIA
senial_iir_t1_f0 = signal.lfilter(Num_iir_t1_f0, Den_iir_t1_f0, senial_a_filtrar)
# #Se aplica el filtrado para tono1 f1 sobre la señal LIMPIA
senial_iir_t1_f1 = signal.lfilter(Num_iir_t1_f1, Den_iir_t1_f1, senial_a_filtrar)
# #Se aplica el filtrado para tono1 f2 sobre la señal LIMPIA
senial_iir_t1_f2 = signal.lfilter(Num_iir_t1_f2, Den_iir_t1_f2, senial_a_filtrar)
#Se aplica el filtrado para tono2 f2
senial_iir_t2_f2 = signal.lfilter(Num_iir_t2_f2, Den_iir_t2_f2, senial_a_filtrar)

#Deberia sumar los resultados de los 4 filtros para tener la señal filtrada
senial_iir = np.zeros(len(senial_a_filtrar))
for i in range(len(senial_a_filtrar)):
    senial_iir[i]= senial_iir_t1_f0[i] + senial_iir_t1_f1[i] + senial_iir_t1_f2[i] + senial_iir_t2_f2[i]

#Hago la fft para cada resultado del filtrado para ver como resulto el filtrado
senial_iir_t1_f0_fft_mod, f_iir_t1_f0 = fftmod(senial_iir_t1_f0, len(senial_iir_t1_f0), fs_a_filtrar)
senial_iir_t1_f1_fft_mod, f_iir_t1_f1 = fftmod(senial_iir_t1_f1, len(senial_iir_t1_f1), fs_a_filtrar)
senial_iir_t1_f2_fft_mod, f_iir_t1_f2 = fftmod(senial_iir_t1_f2, len(senial_iir_t1_f2), fs_a_filtrar)
senial_iir_t2_f2_fft_mod, f_iir_t2_f2 = fftmod(senial_iir_t2_f2, len(senial_iir_t2_f2), fs_a_filtrar)

#===============Graficacion de las señales filtradas===============
if 1==1:
    fig4, ax4 = plt.subplots(2, 4, figsize=(20, 10))    
    
    ax4[0,0].plot(t_a_filtrar, senial_iir_t1_f0, label='Señal filtro t1f0', zorder=1, color='blue')
    ax4[0,0].set_xlabel('Tiempo [s]', fontsize=15)
    ax4[0,0].set_ylabel('Tensión [V]', fontsize=15)
    ax4[0,0].set_title('Señal iir t1f0', fontsize=15)
    # ax4[0,0].set_xlim([0, ts*N])
    ax4[0,0].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax4[0,0].grid()
    ax4[0,0].legend(fontsize=12)
    
    ax4[0,1].plot(t_a_filtrar, senial_iir_t1_f1, label='Señal filtro t1f1', zorder=1, color='blue')
    ax4[0,1].set_xlabel('Tiempo [s]', fontsize=15)
    ax4[0,1].set_ylabel('Tensión [V]', fontsize=15)
    ax4[0,1].set_title('Señal iir t1f1', fontsize=15)
    # ax4[0,1].set_xlim([0, ts*N])
    ax4[0,1].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax4[0,1].grid()
    ax4[0,1].legend(fontsize=12)
    
    ax4[0,2].plot(t_a_filtrar, senial_iir_t1_f2, label='Señal filtro t1f2', zorder=1, color='blue')
    ax4[0,2].set_xlabel('Tiempo [s]', fontsize=15)
    ax4[0,2].set_ylabel('Tensión [V]', fontsize=15)
    ax4[0,2].set_title('Señal iir t1f2', fontsize=15)
    # ax4[0,2].set_xlim([0, ts*N])
    ax4[0,2].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax4[0,2].grid()
    ax4[0,2].legend(fontsize=12)
    
    ax4[0,3].plot(t_a_filtrar, senial_iir_t2_f2, label='Señal filtro t2f2', zorder=1, color='blue')
    ax4[0,3].set_xlabel('Tiempo [s]', fontsize=15)
    ax4[0,3].set_ylabel('Tensión [V]', fontsize=15)
    ax4[0,3].set_title('Señal iir t2f2', fontsize=15)
    # ax4[0,3].set_xlim([0, ts*N])
    ax4[0,3].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax4[0,3].grid()
    ax4[0,3].legend(fontsize=12)
    
    # Se grafica la magnitud de la respuesta en frecuencia de cada filtrado
    ax4[1,0].plot(f_iir_t1_f0, senial_iir_t1_f0_fft_mod)
    ax4[1,0].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax4[1,0].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax4[1,0].set_title('Magnitud frec iir t1f0', fontsize=15)
    # ax4[1,0].set_xlim([0, fs/2])
    ax4[1,0].set_xlim([0, 5000])
    ax4[1,0].grid()
    ax4[1,0].legend(fontsize=12)
    
    ax4[1,1].plot(f_iir_t1_f1, senial_iir_t1_f1_fft_mod)
    ax4[1,1].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax4[1,1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax4[1,1].set_title('Magnitud frec iir t1f1', fontsize=15)
    # ax4[1,1].set_xlim([0, fs/2])
    ax4[1,1].set_xlim([0, 5000])
    ax4[1,1].grid()
    ax4[1,1].legend(fontsize=12)
    
    ax4[1,2].plot(f_iir_t1_f2, senial_iir_t1_f2_fft_mod)
    ax4[1,2].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax4[1,2].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax4[1,2].set_title('Magnitud frec iir t1f2', fontsize=15)
    # ax4[1,2].set_xlim([0, fs/2])
    ax4[1,2].set_xlim([0, 5000])
    ax4[1,2].grid()
    ax4[1,2].legend(fontsize=12)
    
    ax4[1,3].plot(f_iir_t2_f2, senial_iir_t2_f2_fft_mod)
    ax4[1,3].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax4[1,3].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax4[1,3].set_title('Magnitud frec iir t2f2', fontsize=15)
    # ax4[1,3].set_xlim([0, fs/2])
    ax4[1,3].set_xlim([0, 5000])
    ax4[1,3].grid()
    ax4[1,3].legend(fontsize=12)
    
    #Grafica de la señal filtrada por esos 4 filtros
    fig5, ax5 = plt.subplots(1, 1, figsize=(20, 10))    
    
    ax5.plot(t_a_filtrar, senial_iir, label='Señal iir filtrada', zorder=1, color='blue')
    ax5.set_xlabel('Tiempo [s]', fontsize=15)
    ax5.set_ylabel('Tensión [V]', fontsize=15)
    ax5.set_title('Señal iir', fontsize=15)
    # ax5.set_xlim([0, ts*N])
    ax5.set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax5.grid()
    ax5.legend(fontsize=12)

#Genero un .wav de la senial filtrada
senial_iir=senial_iir.astype('float32')
wavfile.write('senial_iir_UTI.wav',fs_a_filtrar,senial_iir)