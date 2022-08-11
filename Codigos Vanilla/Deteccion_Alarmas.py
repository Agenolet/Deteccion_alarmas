# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Detección ALarmas:
    En el siguiente script se ejemplifica el uso del algoritmo de detección de 
    alarmas suministrado por la cátedra.

Autor: Albano Peñalva
Fecha: Mayo 2020

"""

# Librerías
import matplotlib.pyplot as plt
import detector_alarmas
from crear_senial import crear_senial
plt.close('all') # cerrar gráficas anteriores

####################### Generación de señal de prueba ########################

G_NEWPORT = 1
G_PURITAN = 1
G_DRAGGER = 1
G_RUIDO = 0

N_NEWPORT = 1
N_PURITAN = 0
N_DRAGGER = 0

SOLAPAMIENTO = 0

# Se utiliza la función crear_senial()
t, senial = crear_senial(G_NEWPORT, G_PURITAN, G_DRAGGER, G_RUIDO, N_NEWPORT, 
                         N_PURITAN, N_DRAGGER, SOLAPAMIENTO)
ts = t[1] - t[0]
fs = 1 / ts
N = len(senial)

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

############################ Filtrado de la Señal ############################

# En esta instancia se debería aplicar el filtrado correspondiente sobre la 
# señal. Se omite este paso en el presente ejemplo. 
senial_filt = senial

########################### Algoritmo de Detección ###########################

# Se instancia la clase detector_alarmas seleccionando la alarma a detectar
alarm_detect = detector_alarmas.DetectorAlarmas('Newport', fs)
# alarm_detect = detector_alarmas.DetectorAlarmas('Dragger', fs)
# alarm_detect = detector_alarmas.DetectorAlarmas('Puritan', fs)

alarmas = alarm_detect.detectar_alarma(senial_filt)

# Se grafican las alarmas detectadas
for i in range(len(alarmas)):
    ax1.arrow(alarmas[i], 1, 0, -1, color='red',
              width=0.5, length_includes_head=True, zorder=5, 
              head_length=0.15, alpha=0.5)
ax1.legend(loc="lower right", fontsize=15)

plt.show()