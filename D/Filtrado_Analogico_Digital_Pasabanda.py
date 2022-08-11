# -*- coding: utf-8 -*-

"""

Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Filtrado Analógico:
    En el siguiente script se ejemplifica el proceso de diseño de filtros 
    analógicos, el cálculo de componentes para su implementación utilizando 
    filtros activos y el análisis de la simulación de los mismos realizada 
    mediante el software LTSpice.

Autor: Albano Peñalva
Fecha: Mayo 2020

"""

# %% Librerías

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from import_ltspice import import_AC_LTSpice

plt.close('all') # cerrar gráficas anteriores

# %% Definición de requisitos

APROXIMACION = 'Chebyshev'  # 'Chebyshev' o 'Butterworth'
TIPO =  'Pasa Banda'         # 'Pasa Bajo', 'Pasa Alto' o 'Pasa Banda'
RIPPLE = 3                  # en dB (usado en aprox. de Chebychev)
FP1 = 1200                  # Frec. límite para la banda de paso en Hz (frec. de corte en Butterworth, frec. de fin de ripple para Chebyshev)
FS1 = 500                  # Frec. límite para la banda de rechazo en Hz
AT1 = 20                    # Atenuación mínima en dB en la banda de rechazo (a partir de FS1)

if TIPO == 'Pasa Banda':
    FP2 = 2000              # Frec. límite (superior) para la banda de paso en Hz (para pasa banda)
    FS2 = 3000              # Frec. límite (superior) para la banda de rechazo en Hz (para pasa banda)
    AT2 = 20                # Atenuación mínima en dB en la banda de rechazo (a partir de FS2)
    
# %% Normalización de requisitos

# Aplicando la transformación en frecuencia correspondiente, analizamos los 
# requisitos de nuestro filtros respecto al filtro pasa bajops normalizado.

wp1 = 2 * np.pi * FP1   # Convertimos a rad/s
ws1 = 2 * np.pi * FS1
wp2 = 0
ws2 = 0

if TIPO == 'Pasa Bajo':
    ws1_n = ws1 / wp1   # Transformación Pasa Bajos -> Pasa Bajos
    
if TIPO == 'Pasa Alto':
    ws1_n = wp1 / ws1   # Transformación Pasa Bajos -> Pasa Altos
    
if TIPO == 'Pasa Banda':
    wp2 = 2 * np.pi * FP2
    ws2 = 2 * np.pi * FS2
    w0 = np.sqrt(wp1 * wp2) # Frecuencia central en rad/s
    B = wp2 - wp1           # Ancho de banda
    ws1_n = np.abs(ws1 ** 2 - w0 ** 2) / (ws1 * B) # Transformación Pasa Bajos -> Pasa Banda
    ws2_n = np.abs(ws2 ** 2 - w0 ** 2) / (ws2 * B) # Transformación Pasa Bajos -> Pasa Banda

# %% Cálculo del orden óptimo del filtro

w_n = np.logspace(-1, 2, int(10e3))   # vector de frecuencia (en rad/s) para el filtro normalizado
n = 0   # Orden del filtro
at1_n = 0 # Atenuación del filtro normalizado en ws1
at2_n = 0 # Atenuación del filtro normalizado en ws2

if (TIPO == 'Pasa Bajo' or TIPO == 'Pasa Alto'):
    ws2_n = ws1_n
    condicion = lambda at1, at2: at1 < AT1
    
if (TIPO == 'Pasa Banda'):
    condicion = lambda at1, at2: (at1 < AT1) or (at2 < AT2)
    
# Se crea una gráfica para mostrar las distintas iteraciones 
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
ax1.set_title('Filtro '+APROXIMACION+' Normalizado', fontsize=18)
ax1.set_xlabel('Frecuencia [rad/s]', fontsize=15)
ax1.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax1.set_xscale('log')
ax1.set_xlim(0.1, 100)
ax1.grid(True, which="both")
ax1.plot(ws1_n, -AT1, 'X', label='Requisito de atenuación')
if (TIPO == 'Pasa Banda'):
    ax1.plot(ws2_n, -AT2, 'X', label='Requisito de atenuación')
ax1.legend(loc="lower left", fontsize=15)

# Se repite el cilo aumentando el orden del filtro hasta obtener las 
# atenuaciones requeridas
while condicion(at1_n, at2_n):
    n = n + 1 # Aumentamos el orden
    
    # Se obtienen ceros, polos y ganancia 
    if APROXIMACION == 'Butterworth':
        [z_n, p_n, k_n] = signal.buttap(n) 
    if APROXIMACION == 'Chebyshev':
        [z_n, p_n, k_n] = signal.cheb1ap(n, RIPPLE)
        
    # Se obtiene Numerador y Denominador de la función de transferencia del 
    # filtro normalizado de orden n
    [num_n, den_n] = signal.zpk2tf(z_n, p_n, k_n)
    
    # Se calcula la atenuación en las frecuencias de interés para evaluar 
    # atenuación y en todo w_n para graficación
    _, at_n = signal.freqs(num_n, den_n, worN=[ws1_n, ws2_n])
    at1_n = -20 * np.log10(abs(at_n[0]))
    at2_n = -20 * np.log10(abs(at_n[1]))
    _, h_n = signal.freqs(num_n, den_n, worN=w_n)
    
    # Se grafica la respuesta en frecuencia del filtro normalizado de orden n
    ax1.plot(w_n, 20*np.log10(abs(h_n)), label='Orden {}'.format(n))
    ax1.legend(loc="lower left", fontsize=15)

# %% Desnormalización de la Función de Transferencia

# En num_n y den_n se encuentran los coeficientes de la función de transferencia
# normalizada del orden seleccionado. Ahora debemos aplicar la transformación 
# en frecuencia correspondiente para desnormalizarla.

s = sy.Symbol('s') # Se crea una variable simbólica s
s_n = sy.Symbol('s_n') 

if TIPO == 'Pasa Bajo':
    s_n = s / wp1   # Transformación Pasa Bajos -> Pasa Bajos
    
if TIPO == 'Pasa Alto':
    s_n = wp1 / s   # Transformación Pasa Bajos -> Pasa Altos
    
if TIPO == 'Pasa Banda':
    s_n = (s ** 2 + w0 ** 2) / (s * B) # Transformación Pasa Bajos -> Pasa Banda

# Se aplica la transformación correspondinte y se simplifica la expresión
num_s = num_n
den_s = 0
for i in range(n + 1):
    den_s = den_s + sy.expand(den_n[i] * np.power(s_n, n - i))
Hs = num_s * sy.expand(den_s.as_numer_denom()[1]) / sy.expand(den_s.as_numer_denom()[0])
Hs = sy.factor(Hs)[0]

print("La función de transferencia del Filtro desnormalizado es:\r\n")    
print(sy.pretty(Hs.evalf(3))) 
print("\r\n")

# Se extraen coeficientes del numerador y denominador
num = np.array(sy.poly(Hs.as_numer_denom()[0], s).all_coeffs()).astype(float)
den = np.array(sy.poly(Hs.as_numer_denom()[1], s).all_coeffs()).astype(float)

# Se calcula la atenuación en las frecuencias de interés para evaluar 
# atenuación y en todo f para graficación
f = np.logspace(0, 5, int(1e3))
_, at = signal.freqs(num, den, worN=[ws1, ws2])
at1 = -20 * np.log10(abs(at[0]))
at2 = -20 * np.log10(abs(at[1]))
_, h = signal.freqs(num, den, worN=2*np.pi*f)

print("La atenuación del filtro en {}Hz es de {:.2f}dB".format(FS1, at1))
if (TIPO == 'Pasa Banda'):
    print("La atenuación del filtro en {}Hz es de {:.2f}dB".format(FS2, at2))
print('\r\n')

# Se crea una gráfica para mostrar la respuesta en frecuencia del filtro diseñado
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
ax2.set_title('Filtro '+APROXIMACION+' '+TIPO, fontsize=18)
ax2.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax2.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax2.set_xscale('log')
ax2.set_xlim(1, 1e5)
ax2.grid(True, which="both")
ax2.plot(FS1, -AT1, 'X', label='Requisito de atenuación')
ax2.plot(f, 20*np.log10((abs(h))), label=APROXIMACION+' orden {}'.format(n))
if (TIPO == 'Pasa Banda'):
    ax2.plot(FS2, -AT2, 'X', label='Requisito de atenuación')
ax2.legend(loc="lower left", fontsize=15)

# %% Factorización en secciones de orden 2

# Con el objetivo de poder implementarlo usando celdas de Sallen Key o 
# Multiples Realimentaciones se separa la función de transferencia en 
# funciones de orden 2

'''
Esta sección del código no está generalizada para distintos tipos de filtros u
órdenes. Sólo se implementa para un filtro pasa bajos de orden 4. Realizar las 
modificaciones necesarias para implementar otros filtros.
'''

# Se obtienen polos y ceros de la función de transferencia
H = signal.TransferFunction(num, den)
ceros = H.to_zpk().zeros
polos = H.to_zpk().poles
gan = H.to_zpk().gain

# Se separan los polos en pares complejos conjugados
polos_1 = polos[0 : 2]
polos_2 = polos[2 : 4]

# Se separan los ceros (como es un pasabajos la variable ceros está vacía)
ceros_1 = []
ceros_2 = ceros # Se lo hace pasaaltos al segundo, entonces se le asignan los dos ceros (s^2)

# Se separa la ganancia con el criterio de que el Pasa Bajos 1 tenga ganancia unitaria 
gan_1 = (abs(polos_1[0])) ** 2;
gan_2 = gan / gan_1;

# Se obtinen numerador y denominador de ambos filtros
[num_1, den_1] = signal.zpk2tf(ceros_1, polos_1, gan_1)
[num_2, den_2] = signal.zpk2tf(ceros_2, polos_2, gan_2)

# Se calculan las respuestas en magnitud de ambas secciones de orden 2 y se
# grafican junto a la del filtro de orden 4
_, h_1 = signal.freqs(num_1, den_1, worN=2*np.pi*f)
_, h_2 = signal.freqs(num_2, den_2, worN=2*np.pi*f)
ax2.plot(f, 20*np.log10((abs(h_1))), label='Sección 1')
ax2.plot(f, 20*np.log10((abs(h_2))), label='Sección 2')
ax2.legend(loc="lower left", fontsize=15)

# %% Cálculo de componentes Pasa Bajos 1 implementado con Sallen-Key

# Siguiendo el "Mini Tutorial Sallen-Key"

# Se propone el valor de C1
C1_1 = 100e-9 # 100nF

w0_1 = np.sqrt(den_1[2])    # El termino independiente es w0^2
alpha_1 = den_1[1] / w0_1   # El termino que acompaña a s es alpha*w0
H_1 = num_1[0] / den_1[2]   # Numerador = H * w0^2

k_1 = w0_1 * C1_1; 
m_1 = (alpha_1 ** 2) / 4 + (H_1 - 1)

# En Sallen-Key no se pueden implementar filtros con ganancia menor que 1,
# por lo tanto si H es menor o igual a uno se implementa un seguidor de 
# tensión (R3 = cable, R4 = no se coloca)

if (H_1 <= 1): 
    R3_1 = 0
    R4_1 = np.inf 
else:
    # Se propone R3
    R3_1 = 1e3  # 1K
    R4_1 = R3_1 / (H_1 - 1)

C2_1 = m_1 * C1_1
R1_1 = 2 / (alpha_1 * k_1)
R2_1 = alpha_1 / (2 * m_1 * k_1)

print('Los componentes calculados para la sección 1 son:')
print('R1: {:.2e} ohm'.format(R1_1))
print('R2: {:.2e} ohm'.format(R2_1))
print('R3: {:.2e} ohm'.format(R3_1))
print('R4: {:.2e} ohm'.format(R4_1))
print('C1: {:.2e} ohm'.format(C1_1))
print('C2: {:.2e} ohm'.format(C2_1))
print('\r\n')

# Se utilizaran componentes con valores comerciales para la implementación
R1_1_c = 1.8e4
R2_1_c = 1.8e4
R3_1_c = 0
R4_1_c = np.inf
C1_1_c = C1_1
C2_1_c = 6.8e-10
print('Los componentes comerciales para la sección 1 son:')
print('R1: {:.2e} ohm'.format(R1_1_c))
print('R2: {:.2e} ohm'.format(R2_1_c))
print('R3: {:.2e} ohm'.format(R3_1_c))
print('R4: {:.2e} ohm'.format(R4_1_c))
print('C1: {:.2e} ohm'.format(C1_1_c))
print('C2: {:.2e} ohm'.format(C2_1_c))
print('\r\n')

# %% Cálculo de componentes Pasa Bajos 2 implementado con Multiples Realimentaciones

# Siguiendo el "Mini Tutorial Multiples Realimentaciones"

# Se propone el valor de C5
C5_2 = 10e-9 # 10nF

w0_2 = np.sqrt(den_2[2])    # El termino independiente es w0^2
alpha_2 = den_2[1] / w0_2   # El termino que acompaña a s es alpha*w0
H_2 = num_2[0] / den_2[2]   # Numerador = H * w0^2

k_2 = w0_2 * C5_2; 

C2_2 = (4 / alpha_2 ** 2) * (H_2) * C5_2
R1_2 = alpha_2 / (2 * H_2 * k_2)
R3_2 = alpha_2 / (2 * (H_2 + 1) * k_2)
R4_2 = alpha_2 / (2 * k_2)

print('Los componentes calculados para la sección 2 son:')
print('R1: {:.2e} ohm'.format(R1_2))
print('R3: {:.2e} ohm'.format(R3_2))
print('R4: {:.2e} ohm'.format(R4_2))
print('C2: {:.2e} ohm'.format(C2_2))
print('C5: {:.2e} ohm'.format(C5_2))
print('\r\n')

# Se utilizaran componentes con valores comerciales para la implementación
R1_2_c = 2.2e4
R3_2_c = 10e3
R4_2_c = 1.5e4
C2_2_c = 3.3e-8
C5_2_c = C5_2
print('Los componentes comerciales para la sección 2 son:')
print('R1: {:.2e} ohm'.format(R1_2_c))
print('R3: {:.2e} ohm'.format(R3_2_c))
print('R4: {:.2e} ohm'.format(R4_2_c))
print('C2: {:.2e} ohm'.format(C2_2_c))
print('C5: {:.2e} ohm'.format(C5_2_c))
print('\r\n')

# %% Evaluación de resultados de simulación de filtros 

# Luego de simular los filtros utilizando el software LTSpice, se cargan los
# resultados obtenidos para realizar la comparación con el diseño "ideal"

f1, h_sim_1, _ = import_AC_LTSpice('SallenKey_PasaBanda.txt')
#f2, h_sim_2, _ = import_AC_LTSpice('MFB_PasaBajos.txt')


# Analógico a digital
# Se aplica transformada bilineal
num_D, den_D = signal.bilinear(num_2, den_2, 44100)
# Se encuentra transformada z
f_D, h_D = signal.freqz(num_D,den_D,worN=f1, fs = 44100)
# Se expresa h_D en dB
h_D = 20*np.log10(abs(h_D))


# Se crea una gráfica para comparar los filtros 
fig3, ax3 = plt.subplots(2, 2, figsize=(12, 12))

ax3[0, 0].set_title('Sección 1', fontsize=18)
ax3[0, 0].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3[0, 0].set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3[0, 0].set_xscale('log')
ax3[0, 0].set_xlim(1, 1e5)
ax3[0, 0].grid(True, which="both")
ax3[0, 0].plot(f, 20*np.log10((abs(h_1))), label='Ideal')
ax3[0, 0].plot(f1, h_sim_1, label='Simulado')
ax3[0, 0].legend(loc="lower left", fontsize=15)

ax3[0, 1].set_title('Sección 2', fontsize=18)
ax3[0, 1].set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3[0, 1].set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3[0, 1].set_xscale('log')
ax3[0, 1].set_xlim(1, 1e5)
ax3[0, 1].grid(True, which="both")
ax3[0, 1].plot(f, 20*np.log10((abs(h_2))), label='Ideal')
ax3[0, 1].plot(f_D, h_D, label='Digital')
ax3[0, 1].set_xlim([0, 44100/2])
ax3[0, 1].legend(loc="lower left", fontsize=15)

gs = ax3[1, 0].get_gridspec()
ax3[1, 0].remove()
ax3[1, 1].remove()
ax3_big = fig3.add_subplot(gs[1, :])
ax3_big.set_title('Filtro '+APROXIMACION+' orden {}'.format(n), fontsize=18)
ax3_big.set_xlabel('Frecuencia [Hz]', fontsize=15)
ax3_big.set_ylabel('|H(jw)|² [dB]', fontsize=15)
ax3_big.set_xscale('log')
ax3_big.set_xlim(1, 1e5)
ax3_big.grid(True, which="both")
ax3_big.plot(f, 20*np.log10((abs(h))), label='Ideal')
ax3_big.plot(f1, h_sim_1 + h_D, label='Simulado + Digital')
ax3_big.set_xlim([0, 44100/2])
ax3_big.legend(loc="lower left", fontsize=15)
plt.tight_layout()