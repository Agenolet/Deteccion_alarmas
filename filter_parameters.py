# -*- coding: utf-8 -*-
"""
Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

Módulo que contiene funciones para el cálculo de la FFT, pensado para 
ejemplificar el uso de funciones en Python.

Autor: JC
Fecha: Mayo 2020
"""
import numpy as np


def filter_parameters(fil):
    """
    ------------------------
    INPUT:
    --------
    fil: archivo .npz generado con pyFDA
    ------------------------
    OUTPUT:
    --------
    nada, devuelve por consola los parámetros principales    
    """
    fil = np.load(fil, allow_pickle=True)
    print("Frecuencia de muestreo: {:.1f} Hz".format(fil['f_S']))   
    print("Aproximación utilizada:" + fil['creator'][1])
    print("Orden del filtro: {:}".format(fil['N']))  
    tipo = fil['rt']
    if tipo == 'LP':
        print("Filtro Pasa-Bajos")
        print("Frecuencia de corte banda de paso : {:.1f} Hz".format(fil['F_PB']*fil['f_S']))
        print("Frecuencia de corte banda de rechazo: {:.1f} Hz".format(fil['F_SB']*fil['f_S']))
        
    if tipo == 'BP':
        print("Filtro Pasa-Banda")
        print("Frecuencia de corte banda de rechazo inferior: {:.1f} Hz".format(fil['F_SB']*fil['f_S']))
        print("Frecuencia de corte inferior banda de paso : {:.1f} Hz".format(fil['F_PB']*fil['f_S']))
        print("Frecuencia de corte superior banda de paso : {:.1f} Hz".format(fil['F_PB2']*fil['f_S']))
        print("Frecuencia de corte banda de rechazo superior: {:.1f} Hz".format(fil['F_SB2']*fil['f_S']))

    if tipo == 'BS':
        print("Filtro Rechaza-Banda")
        print("Frecuencia de corte banda de paso inferior: {:.1f} Hz".format(fil['F_PB']*fil['f_S']))
        print("Frecuencia de corte inferior banda de rechazo : {:.1f} Hz".format(fil['F_SB']*fil['f_S']))
        print("Frecuencia de corte superior banda de rechazo : {:.1f} Hz".format(fil['F_SB2']*fil['f_S']))
        print("Frecuencia de corte banda de paso superior: {:.1f} Hz".format(fil['F_PB2']*fil['f_S']))
        
    if tipo == 'HP':
        print("Filtro Pasa-Alto")
        print("Frecuencia de corte banda de paso : {:.1f} Hz".format(fil['F_SB']*fil['f_S']))
        print("Frecuencia de corte banda de rechazo: {:.1f} Hz".format(fil['F_PB']*fil['f_S']))

#ejemplo
#filter_parameters('filtro_iir.npz')