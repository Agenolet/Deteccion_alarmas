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
from scipy import signal
from time import time
from filter_parameters import filter_parameters
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

############################### Etapa 2: Filtros #############################  
#Se trabaja para distingir la Dragger_Carina del resto de alarmas, y de un ambiente ruidoso en general

# FILTROS IIR
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

# FILTROS FIR
#Cargo el filtro para tono 1 - f0
filtro_fir_t1_f0 = np.load('./Filtros/' + 'Dragger_Carina_t1_f0_FIR.npz', allow_pickle=True)
Num_fir_t1_f0, Den_fir_t1_f0 = filtro_fir_t1_f0['ba'] 
# #Cargo el filtro para tono 1 - f1
filtro_fir_t1_f1 = np.load('./Filtros/' + 'Dragger_Carina_t1_f1_FIR.npz', allow_pickle=True)
Num_fir_t1_f1, Den_fir_t1_f1 = filtro_fir_t1_f1['ba'] 
# #Cargo el filtro para tono 1 - f2
filtro_fir_t1_f2 = np.load('./Filtros/' + 'Dragger_Carina_t1_f2_FIR.npz', allow_pickle=True)
Num_fir_t1_f2, Den_fir_t1_f2 = filtro_fir_t1_f2['ba'] 
# #Cargo el filtro para tono 2 - f2
filtro_fir_t2_f2 = np.load('./Filtros/' + 'Dragger_Carina_t2_f2_FIR.npz', allow_pickle=True)
Num_fir_t2_f2, Den_fir_t2_f2 = filtro_fir_t2_f2['ba'] 


#Genero una señal como la suma de varias señales de alarmas. El if es para poder desplejar el codigo y acortarlo
if 1==1:
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
    senial_sumada = np.zeros(largo_menor)
    for i in range(largo_menor):
    	senial_sumada[i] = senial_n[0][i]+senial_n[1][i]+senial_n[2][i]+senial_n[3][i]

    fs_sumada = fs_n[0]
    N_senial_sumada = largo_menor 
    # Genero un .wav de la senial SUMADA
    # senial_sumada_wav=senial_sumada.astype('int16')
    # wavfile.write('Senial_Sumada_int.wav',fs_sumada,senial_sumada)
    

#Cargo la señal que quiero filtrar

# filename_a_filtrar = 'UTI.wav'      
# filename_a_filtrar = 'Dragger_Carina.wav'  
# filename_a_filtrar = 'Newport_HT50.wav'  
# filename_a_filtrar = 'Puritan_Bennett_840.wav'         
# filename_a_filtrar = 'Dragger_Carina_UTI.wav' 
# filename_a_filtrar = 'Senial_Sumada_int.wav' 
         
# fs_a_filtrar, data_a_filtrar = wavfile.read('./Alarmas/' + filename_a_filtrar)   
# ts_a_filtrar = 1 / fs_a_filtrar                   
# N_a_filtrar = len(data_a_filtrar)               
# t_a_filtrar = np.linspace(0, N_a_filtrar * ts_a_filtrar, N_a_filtrar)  
# senial_a_filtrar = data_a_filtrar[:, 1] 
# senial_a_filtrar = senial_a_filtrar * 3.3 / (2 ** 16 - 1)

# Si cargo la señal sumada:
senial_a_filtrar = senial_sumada
fs_a_filtrar = fs_n[0]
ts_a_filtrar = 1 / fs_a_filtrar
N_a_filtrar = N_senial_sumada
t_a_filtrar = np.linspace(0, N_a_filtrar * ts_a_filtrar, N_a_filtrar) 

# # Graficacion del espectro de frecuencias de la señal sumada
# senial_sumada_fft_mod, f_senial_sumada = fftmod(senial_sumada, len(senial_sumada), fs_a_filtrar)

# fig8, ax8 = plt.subplots(1, 1, figsize=(20, 10))    
    
# ax8.plot(f_senial_sumada, senial_sumada_fft_mod, label='fftmod senial sumada', zorder=1, color='blue')
# ax8.set_xlabel('Tiempo [s]', fontsize=15)
# ax8.set_ylabel('Tensión [V]', fontsize=15)
# ax8.set_title('fftmod senial sumada', fontsize=15)
# # ax8.set_xlim([0, fs_a_filtrar/2])
# ax8.set_xlim([0, 9000]) 
# ax8.set_ylim([0, 0.02])
# ax8.grid()
# ax8.legend(fontsize=12)

#===============Aplicacion del filtro IIR ===============
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

#Hago la fft para cada resultado del filtrado para ver como resultó el filtrado
senial_iir_t1_f0_fft_mod, f_iir_t1_f0 = fftmod(senial_iir_t1_f0, len(senial_iir_t1_f0), fs_a_filtrar)
senial_iir_t1_f1_fft_mod, f_iir_t1_f1 = fftmod(senial_iir_t1_f1, len(senial_iir_t1_f1), fs_a_filtrar)
senial_iir_t1_f2_fft_mod, f_iir_t1_f2 = fftmod(senial_iir_t1_f2, len(senial_iir_t1_f2), fs_a_filtrar)
senial_iir_t2_f2_fft_mod, f_iir_t2_f2 = fftmod(senial_iir_t2_f2, len(senial_iir_t2_f2), fs_a_filtrar)


#===============Aplicacion del filtro FIR ===============
#Se aplica el filtrado para f0 sobre la señal LIMPIA
senial_fir_t1_f0 = signal.lfilter(Num_fir_t1_f0, Den_fir_t1_f0, senial_a_filtrar)
# #Se aplica el filtrado para tono1 f1 sobre la señal LIMPIA
senial_fir_t1_f1 = signal.lfilter(Num_fir_t1_f1, Den_fir_t1_f1, senial_a_filtrar)
# #Se aplica el filtrado para tono1 f2 sobre la señal LIMPIA
senial_fir_t1_f2 = signal.lfilter(Num_fir_t1_f2, Den_fir_t1_f2, senial_a_filtrar)
#Se aplica el filtrado para tono2 f2
senial_fir_t2_f2 = signal.lfilter(Num_fir_t2_f2, Den_fir_t2_f2, senial_a_filtrar)

#Deberia sumar los resultados de los 4 filtros para tener la señal filtrada
senial_fir = np.zeros(len(senial_a_filtrar))
for i in range(len(senial_a_filtrar)):
    senial_fir[i]= senial_fir_t1_f0[i] + senial_fir_t1_f1[i] + senial_fir_t1_f2[i] + senial_fir_t2_f2[i]

#Hago la fft para cada resultado del filtrado para ver como resultó el filtrado
senial_fir_t1_f0_fft_mod, f_fir_t1_f0 = fftmod(senial_fir_t1_f0, len(senial_fir_t1_f0), fs_a_filtrar)
senial_fir_t1_f1_fft_mod, f_fir_t1_f1 = fftmod(senial_fir_t1_f1, len(senial_fir_t1_f1), fs_a_filtrar)
senial_fir_t1_f2_fft_mod, f_fir_t1_f2 = fftmod(senial_fir_t1_f2, len(senial_fir_t1_f2), fs_a_filtrar)
senial_fir_t2_f2_fft_mod, f_fir_t2_f2 = fftmod(senial_fir_t2_f2, len(senial_fir_t2_f2), fs_a_filtrar)

#===============Graficacion de las señales filtradas por IIR===============
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


#===============Graficacion de las señales filtradas por FIR===============
if 1==1:
    fig5, ax5 = plt.subplots(2, 4, figsize=(20, 10))    
    
    ax5[0,0].plot(t_a_filtrar, senial_fir_t1_f0, label='Señal filtroFIR t1f0', zorder=1, color='blue')
    ax5[0,0].set_xlabel('Tiempo [s]', fontsize=15)
    ax5[0,0].set_ylabel('Tensión [V]', fontsize=15)
    ax5[0,0].set_title('Señal fir t1f0', fontsize=15)
    # ax5[0,0].set_xlim([0, ts*N])
    ax5[0,0].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax5[0,0].grid()
    ax5[0,0].legend(fontsize=12)
    
    ax5[0,1].plot(t_a_filtrar, senial_fir_t1_f1, label='Señal filtro t1f1', zorder=1, color='blue')
    ax5[0,1].set_xlabel('Tiempo [s]', fontsize=15)
    ax5[0,1].set_ylabel('Tensión [V]', fontsize=15)
    ax5[0,1].set_title('Señal fir t1f1', fontsize=15)
    # ax5[0,1].set_xlim([0, ts*N])
    ax5[0,1].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax5[0,1].grid()
    ax5[0,1].legend(fontsize=12)
    
    ax5[0,2].plot(t_a_filtrar, senial_fir_t1_f2, label='Señal filtro t1f2', zorder=1, color='blue')
    ax5[0,2].set_xlabel('Tiempo [s]', fontsize=15)
    ax5[0,2].set_ylabel('Tensión [V]', fontsize=15)
    ax5[0,2].set_title('Señal fir t1f2', fontsize=15)
    # ax5[0,2].set_xlim([0, ts*N])
    ax5[0,2].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax5[0,2].grid()
    ax5[0,2].legend(fontsize=12)
    
    ax5[0,3].plot(t_a_filtrar, senial_fir_t2_f2, label='Señal filtro t2f2', zorder=1, color='blue')
    ax5[0,3].set_xlabel('Tiempo [s]', fontsize=15)
    ax5[0,3].set_ylabel('Tensión [V]', fontsize=15)
    ax5[0,3].set_title('Señal fir t2f2', fontsize=15)
    # ax5[0,3].set_xlim([0, ts*N])
    ax5[0,3].set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax5[0,3].grid()
    ax5[0,3].legend(fontsize=12)
    
    # Se grafica la magnitud de la respuesta en frecuencia de cada filtrado
    ax5[1,0].plot(f_fir_t1_f0, senial_fir_t1_f0_fft_mod)
    ax5[1,0].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax5[1,0].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax5[1,0].set_title('Magnitud frec fir t1f0', fontsize=15)
    # ax5[1,0].set_xlim([0, fs/2])
    ax5[1,0].set_xlim([0, 5000])
    ax5[1,0].grid()
    ax5[1,0].legend(fontsize=12)
    
    ax5[1,1].plot(f_fir_t1_f1, senial_fir_t1_f1_fft_mod)
    ax5[1,1].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax5[1,1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax5[1,1].set_title('Magnitud frec fir t1f1', fontsize=15)
    # ax5[1,1].set_xlim([0, fs/2])
    ax5[1,1].set_xlim([0, 5000])
    ax5[1,1].grid()
    ax5[1,1].legend(fontsize=12)
    
    ax5[1,2].plot(f_fir_t1_f2, senial_fir_t1_f2_fft_mod)
    ax5[1,2].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax5[1,2].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax5[1,2].set_title('Magnitud frec fir t1f2', fontsize=15)
    # ax5[1,2].set_xlim([0, fs/2])
    ax5[1,2].set_xlim([0, 5000])
    ax5[1,2].grid()
    ax5[1,2].legend(fontsize=12)
    
    ax5[1,3].plot(f_fir_t2_f2, senial_fir_t2_f2_fft_mod)
    ax5[1,3].set_xlabel('Frecuencia [Hz]', fontsize=15)
    ax5[1,3].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
    ax5[1,3].set_title('Magnitud frec fir t2f2', fontsize=15)
    # ax5[1,3].set_xlim([0, fs/2])
    ax5[1,3].set_xlim([0, 5000])
    ax5[1,3].grid()
    ax5[1,3].legend(fontsize=12)
    
    #Grafica de la señal filtrada por esos 4 filtros
    fig6, ax6 = plt.subplots(1, 1, figsize=(20, 10))    
    
    ax6.plot(t_a_filtrar, senial_fir, label='Señal fir filtrada', zorder=1, color='blue')
    ax6.set_xlabel('Tiempo [s]', fontsize=15)
    ax6.set_ylabel('Tensión [V]', fontsize=15)
    ax6.set_title('Señal fir', fontsize=15)
    # ax6.set_xlim([0, ts*N])
    ax6.set_xlim([0, ts_a_filtrar*(N_a_filtrar-4*N_a_filtrar/5)]) #Para graficar solo una "melodia"
    ax6.grid()
    ax6.legend(fontsize=12)


#=========== Genero wav para escuchar el resultado =========== 
#Genero un .wav de la senial filtrada IIR
senial_iir=senial_iir.astype('float32')
wavfile.write('senial_iir_Sumada.wav',fs_a_filtrar,senial_iir)

#Genero un .wav de la senial filtrada FIR
senial_fir=senial_fir.astype('float32')
wavfile.write('senial_fir_Sumada.wav',fs_a_filtrar,senial_fir)

#=============== Evaluar Performance del Filtro ===============
"""
# Se aplica el filtrado sobre la señal 100 veces y se mide el tiempo requerido
# por el algoritmo
t_start_iir = time()
for i in range(10):
    senial_iir_t1_f0_perfonmance = signal.lfilter(Num_iir_t1_f0, Den_iir_t1_f0, senial_a_filtrar)
    senial_iir_t1_f1_perfonmance = signal.lfilter(Num_iir_t1_f1, Den_iir_t1_f1, senial_a_filtrar)
    senial_iir_t1_f2_perfonmance = signal.lfilter(Num_iir_t1_f2, Den_iir_t1_f2, senial_a_filtrar)
    senial_iir_t2_f2_perfonmance = signal.lfilter(Num_iir_t2_f2, Den_iir_t2_f2, senial_a_filtrar)
  
t_end_iir = time()
t_start_fir = time()

for i in range(10):
    senial_fir_t1_f0_perfonmance = signal.lfilter(Num_fir_t1_f0, Den_fir_t1_f0, senial_a_filtrar)
    senial_fir_t1_f1_perfonmance = signal.lfilter(Num_fir_t1_f1, Den_fir_t1_f1, senial_a_filtrar)
    senial_fir_t1_f2_perfonmance = signal.lfilter(Num_fir_t1_f2, Den_fir_t1_f2, senial_a_filtrar)
    senial_fir_t2_f2_perfonmance = signal.lfilter(Num_fir_t2_f2, Den_fir_t2_f2, senial_a_filtrar)

t_end_fir = time()

print("El algoritmo de filtrado FIR toma {:.3f}s".format(t_end_fir - t_start_fir))
print("El algoritmo de filtrado IIR toma {:.3f}s".format(t_end_iir - t_start_iir))
relacion=(t_end_fir - t_start_fir) / (t_end_iir - t_start_iir)
print("Son unas {:.3f}s veces".format(relacion))
"""
#=============== Parametros de los Filtros ===============
"""
print('======IIR=====')
print('==Filtro t1f0==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t1_f0.npz')
print('==Filtro t1f1==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t1_f1.npz')
print('==Filtro t1f2==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t1_f2.npz')
print('==Filtro t2f2==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t2_f2.npz')
print('======FIR=====')
print('==Filtro t1f0==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t1_f0_FIR.npz')
print('==Filtro t1f1==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t1_f1_FIR.npz')
print('==Filtro t1f2==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t1_f2_FIR.npz')
print('==Filtro t2f2==')
filter_parameters('./Filtros/' + 'Dragger_Carina_t2_f2_FIR.npz')
"""
# print('======IIR=====')
zeros_iir_t1f0, polos_iir_t1f0, k_iir_t1f0 = filtro_iir_t1_f0['zpk']
for n in range(0,len(polos_iir_t1f0)):
    if np.sqrt(round(polos_iir_t1f0[n].real,4)**2 + round(polos_iir_t1f0[n].imag,4)**2)>=1:
        print('======IIR=====')
        print('==Filtro t1f0==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

zeros_iir_t1f1, polos_iir_t1f1, k_iir_t1f1 = filtro_iir_t1_f1['zpk']
for n in range(0,len(polos_iir_t1f1)):
    if np.sqrt(round(polos_iir_t1f1[n].real,4)**2 + round(polos_iir_t1f1[n].imag,4)**2)>=1:
        print('==Filtro t1f1==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

zeros_iir_t1f2, polos_iir_t1f2, k_iir_t1f2 = filtro_iir_t1_f2['zpk']
for n in range(0,len(polos_iir_t1f2)):
    if np.sqrt(round(polos_iir_t1f2[n].real,4)**2 + round(polos_iir_t1f2[n].imag,4)**2)>=1:
        print('==Filtro t1f2==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

zeros_iir_t2f2, polos_iir_t2f2, k_iir_t2f2 = filtro_iir_t2_f2['zpk']
for n in range(0,len(polos_iir_t2f2)):
    if np.sqrt(round(polos_iir_t2f2[n].real,4)**2 + round(polos_iir_t2f2[n].imag,4)**2)>=1:
        print('==Filtro t2f2==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

# print('======FIR=====')
zeros_fir_t1f0, polos_fir_t1f0, k_fir_t1f0 = filtro_fir_t1_f0['zpk']
for n in range(0,len(polos_fir_t1f0)):
    if np.sqrt(round(polos_fir_t1f0[n].real,4)**2 + round(polos_fir_t1f0[n].imag,4)**2)>=1:
        print('======FIR=====')
        print('==Filtro t1f0==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

zeros_fir_t1f1, polos_fir_t1f1, k_fir_t1f1 = filtro_fir_t1_f1['zpk']
for n in range(0,len(polos_fir_t1f1)):
    if np.sqrt(round(polos_fir_t1f1[n].real,4)**2 + round(polos_fir_t1f1[n].imag,4)**2)>=1:
        print('==Filtro t1f1==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')


zeros_fir_t1f2, polos_fir_t1f2, k_fir_t1f2 = filtro_fir_t1_f2['zpk']
for n in range(0,len(polos_fir_t1f2)):
    if np.sqrt(round(polos_fir_t1f2[n].real,4)**2 + round(polos_fir_t1f2[n].imag,4)**2)>=1:
        print('==Filtro t1f2==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

zeros_fir_t2f2, polos_fir_t2f2, k_fir_t2f2 = filtro_fir_t2_f2['zpk']
for n in range(0,len(polos_fir_t2f2)):
    if np.sqrt(round(polos_fir_t2f2[n].real,4)**2 + round(polos_fir_t2f2[n].imag,4)**2)>=1:
        print('==Filtro t2f2==')
        print('El sistema puede ser inestable si se redondea con 3 decimales')

#=============== Relacion Señal-Ruido para ambos filtrados ===============
# Se comparará senial_iir contra senial_fir. Las frecuencias caracteristicas seran las mismas, ya que la "señal" es la misma, la de Dragger.

#Se calcula espectro de potencia

# PORQUE TENGO QUE HACER AL DELTA LA MITAD? O DICHO DE OTRA FORMA,
# PORQUE TENGO SUPONER QUE TIENE EL DOBLE DE LONGITUD?
# ¿ES POR EL RECORTE DE LAS FRECUENCIAS NEGATIVAS?

f_iir, senial_iir_fft_pot = funciones_fft.fft_pot(senial_iir, fs_a_filtrar)
delta_f_iir = fs_a_filtrar/(len(f_iir)*2)

f_fir, senial_fir_fft_pot = funciones_fft.fft_pot(senial_fir, fs_a_filtrar)
delta_f_fir = fs_a_filtrar/(len(f_fir)*2)

# pot_ruido_iir = senial_iir_fft_pot
# pot_ruido_fir = senial_fir_fft_pot

#Voy a considerar señal, a la compuesta por las frecuencias filtradas: t1f0 t1f1 t1f2 y t2f2
frec_interes = (991, 1982, 3964, 2359)
for i in range(0,4):
    pot_frec_iir = np.sum( senial_iir_fft_pot[int((frec_interes[i]-10)/delta_f_iir) : int((frec_interes[i]+10)/delta_f_iir)] )
    pot_frec_fir = np.sum( senial_fir_fft_pot[int((frec_interes[i]-10)/delta_f_fir) : int((frec_interes[i]+10)/delta_f_fir)] )

pot_ruido_iir = np.sum(senial_iir_fft_pot) - pot_frec_iir
pot_ruido_fir = np.sum(senial_fir_fft_pot) - pot_frec_fir

snr_iir = 10 * np.log10(pot_frec_iir / pot_ruido_iir)
snr_fir = 10 * np.log10(pot_frec_fir / pot_ruido_fir)

print(f"La Relación Señal Ruido con IIR es de {snr_iir:.2f} dB")
print(f"La Relación Señal Ruido con FIR es de {snr_fir:.2f} dB")
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# Grafico los espectros de potencia y señalo donde estan las fracuencias que estoy sumando
fig7, ax7 = plt.subplots(2, 1, figsize=(20, 10))    
ax7[0].plot(f_iir, senial_iir_fft_pot)
for i in range(0,4):
    ax7[0].plot(f_iir[int((frec_interes[i]-10)/delta_f_iir) : int((frec_interes[i]+10)/delta_f_iir) ] , senial_iir_fft_pot[int((frec_interes[i]-10)/delta_f_iir) : int((frec_interes[i]+10)/delta_f_iir) ], color='orange')
ax7[0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax7[0].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
ax7[0].set_title('senial_iir_fft_pot', fontsize=15)
ax7[0].set_xlim([0, fs_a_filtrar/2])
# ax7[0].set_xlim([0, 4500])
ax7[0].grid()

ax7[1].plot(f_fir, senial_fir_fft_pot)
for i in range(0,4):
    ax7[1].plot(f_fir[int((frec_interes[i]-10)/delta_f_fir) : int((frec_interes[i]+10)/delta_f_fir) ] , senial_fir_fft_pot[int((frec_interes[i]-10)/delta_f_fir) : int((frec_interes[i]+10)/delta_f_fir) ], color='orange')
ax7[1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax7[1].set_ylabel('Magnitud [V/√Hz]', fontsize=15)
ax7[1].set_title('senial_fir_fft_pot', fontsize=15)
ax7[1].set_xlim([0, fs_a_filtrar/2])
# ax7[1].set_xlim([0, 4500])
ax7[1].grid()

        
    
    
    
    
    