#!/usr/bin/env python
# coding: utf-8

import csv
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataframe=pd.read_csv('alldataframes.csv') 
from scipy.stats import ttest_ind, mannwhitneyu

# Se hacen los histogramas de cada índice

# Ciclos respiratorios para crepitancias
enfermosconcrep=dataframe[(dataframe.crepitancias == 1)]  # selecciono el grupo que necesito para trabajar
plt.subplots_adjust( wspace=0.6, hspace=0.6)
plt.subplot(2,2,1)
enfermosconcrep.varianza.hist() # .hist me da el histograma 
plt.title('Varianza Crepitancias')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,2)
enfermosconcrep.rango.hist()
plt.title('Rango Crepitancias')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,3)
enfermosconcrep.suma.hist()
plt.title('SMA Crepitancias')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,4)
enfermosconcrep.prom.hist()
plt.title('Media del Espectro Crepitancias')
plt.xlabel('valor')
plt.ylabel('cantidad')

# ciclos respiratorios para sibilancias
enfermosconsib=dataframe[(dataframe.sibilancias == 1)]
plt.figure()
plt.subplots_adjust( wspace=0.6, hspace=0.6)
plt.subplot(2,2,1)
enfermosconsib.varianza.hist()
plt.title('Varianza Sibilancias')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,2)
enfermosconsib.rango.hist()
plt.title('Rango Sibilancias')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,3)
enfermosconsib.suma.hist()
plt.title('SMA Sibilancias')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,4)
enfermosconsib.prom.hist()
plt.title('Media del Espectro Sibilancias')
plt.xlabel('valor')
plt.ylabel('cantidad')

#ciclos respiratorio para crepitancias y sibilancias
enfermosambas=dataframe[(dataframe.crepitancias == 1) & (dataframe.sibilancias == 1)]
plt.figure()
plt.subplots_adjust( wspace=0.6, hspace=0.6)
plt.subplot(2,2,1)
enfermosambas.varianza.hist()
plt.title('Varianza Crep. y Sib.')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,2)
enfermosambas.rango.hist()
plt.title('Rango Crep. y Sib.')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,3)
enfermosambas.suma.hist()
plt.title('SMA Crep. y Sib.')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,4)
enfermosambas.prom.hist()
plt.title(' Media del Espectro Crep. y Sib.')
plt.xlabel('valor')
plt.ylabel('cantidad')

#ciclos respiratorios para sanos
sanos=dataframe[(dataframe.crepitancias == 0) & (dataframe.sibilancias == 0)]
plt.figure()
plt.subplots_adjust( wspace=0.6, hspace=0.6)
plt.subplot(2,2,1)
sanos.varianza.hist()
plt.title(' Varianza Sanos')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,2)
sanos.rango.hist()
plt.title(' Rango Sanos')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,3)
sanos.suma.hist()
plt.title(' SMA Sanos')
plt.xlabel('valor')
plt.ylabel('cantidad')
plt.subplot(2,2,4)
sanos.prom.hist()
plt.title(' Media del Espectro Sanos')
plt.xlabel('valor')
plt.ylabel('cantidad')

#%%

# Hipotesis nula :  El índice evaluado no difieren de manera significativa
# Hipotesis alternativa: El índice evaluado difieren de manera significativa

# Indice a analizar: Rango y media del espectro 

## CREPITANCIAS VS  SIBILANCIAS

# Para rango (según el histograma el comportamiento de ambos es una disribucion normal - prueba t)
crepran=enfermosconcrep['rango'] # selecciono la columna con los datos que necesito
crepran_1=np.array(crepran) # vuelvo la columna vector
sibran=enfermosconsib['rango']
sibran_1=np.array(sibran)
statistics , pvalues = ttest_ind(crepran_1,sibran_1)  # prueba T para distribución normal
print('Valor p para el rango entre ciclos crepitancias vs ciclos sibilancias: ',pvalues)

# Para media del espectro(según el histograma no tiene una distribución normal - test U de Mann Whitney) 
cremed=enfermosconcrep['prom'] # selecciono la columna con los datos que necesito
cremed_1=np.array(cremed)# vuelvo la columna vector
simed=enfermosconsib['prom']
simen_1=np.array(simed)
statistic, pvalue = mannwhitneyu(cremed_1,simen_1) # prueba U de Mann Whitney para distribución no normal
print('Valor p para la media del espectro entre ciclos crepitancias vs ciclos sibilancias: ',pvalue)




