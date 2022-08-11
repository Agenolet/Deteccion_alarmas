import matplotlib.pyplot as plt
import detector_alarmas
from crear_senial import crear_senial
from scipy import signal
import numpy as np
plt.close('all') # cerrar gráficas anteriores

####################### Generación de señal de prueba ########################

G_NEWPORT = 1
G_PURITAN = 1
G_DRAGGER = 0.45
G_RUIDO = 0.3

N_NEWPORT = 0
N_PURITAN = 0
N_DRAGGER = 6

SOLAPAMIENTO = 0

# Se utiliza la función crear_senial()
t, senial = crear_senial(G_NEWPORT, G_PURITAN, G_DRAGGER, G_RUIDO, N_NEWPORT, 
                         N_PURITAN, N_DRAGGER, SOLAPAMIENTO)
ts = t[1] - t[0]
fs = 1 / ts
N = len(senial)

############################ Filtrado de la Señal ############################

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

#===============Aplicacion del filtro IIR ===============
senial_a_filtrar=senial
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

senial_filt = senial_iir

####################### Graficación señal temporal ###########################

# Se crea una gráfica 
fig1, ax1 = plt.subplots(1, 1, figsize=(15, 8))
fig1.suptitle("Señal de audio", fontsize=18)

# Se grafica la señal
ax1.plot(t, senial, label='Señal')
ax1.set_ylabel('Tensión [V]', fontsize=15)
ax1.grid()
ax1.legend(loc="lower right", fontsize=15)
ax1.set_xlim([0, N * ts])
ax1.set_xlabel('Tiempo [s]', fontsize=15)

########################### Algoritmo de Detección ###########################

# Se instancia la clase detector_alarmas seleccionando la alarma a detectar
# alarm_detect = detector_alarmas.DetectorAlarmas('Newport', fs)
alarm_detect = detector_alarmas.DetectorAlarmas('Dragger', fs)
# alarm_detect = detector_alarmas.DetectorAlarmas('Puritan', fs)

alarmas = alarm_detect.detectar_alarma(senial_filt)

# Se grafican las alarmas detectadas
for i in range(len(alarmas)):
    ax1.arrow(alarmas[i], 1, 0, -1, color='red',
              width=0.5, length_includes_head=True, zorder=5, 
              head_length=0.15, alpha=0.5)
ax1.legend(loc="lower right", fontsize=15)

plt.show()