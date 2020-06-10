#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from linearFIR import filter_design, mfreqz
import scipy.signal as signal
import pywt
from scipy.io.wavfile import write
import pandas as pd
import scipy.signal as signal
import os
import sys
from os import listdir
from os.path import isfile, isdir, join
#Se llaman todas las funciones necesarias 


# ## Para el punto 2 :Función para el señal 
# 

# ### Preprocesamiento

# In[8]:


#la función recibe como parametros de entrada la señal y la frecuencia de muestreo que se adquiere tras cargar unsa señal.wav
filename = '226_1b1_Pl_sc_LittC2SE.wav'
senal, Fmuestreo = librosa.load(filename)
def filtradofir(senal,Fmuestreo):
    order, lowpass = filter_design(Fmuestreo, locutoff = 0, hicutoff = 1000, revfilt = 0);
    order, highpass = filter_design(Fmuestreo, locutoff = 100, hicutoff = 0, revfilt = 1);
    #Se definen las frecuencias bajas y altas para el diseño de el filtro pasabanda
    y_hp = signal.filtfilt(highpass, 1, senal);
    y_bp = signal.filtfilt(lowpass, 1, y_hp);
    #función filt filt
    y_bp = np.asfortranarray(y_bp)
    return y_bp , Fmuestreo
#se llama la función
senal_filtrada,sr=filtradofir(senal,Fmuestreo)


# Se usan los filtros lineales FIR que se utilizaron para realizar la practica de laboratorio correspondiente a filtros, para las frecuencias de  interes se plantearon estas de 100 a 1000 Hz , esto se apoyo de :

# In[10]:


plt.figure(figsize=(15,5))

librosa.display.waveplot(senal, Fmuestreo,label='Original');
librosa.display.waveplot(senal_filtrada, Fmuestreo,label='FIR Pasa banda');
plt.ylabel('Amplitude')
plt.title('Comparación señal original y filtrada por Filtro pasa banda')
plt.grid()
plt.legend()
senal_for_wav=librosa.output.write_wav('senalq_for_wav.wav', senal_filtrada, sr)
#se realizan las graficas pertinentes para mostrar el filtrado, además se sobreponen la señal original y la sobrepuesta para 
#garantizar que no exista desfase entre estas.


# ## Para el punto 3: Filtro Wavelet

# In[11]:


#funciones dada por el profesor para la implementación del wavelet
#se defininen los coeficientes relacionados con las aproximaciones y las reconstrucciones
def wthresh(coeff,thr):
    y   = list();
    s = wnoisest(coeff);
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    return y;

#threshold
def thselect(signal):
    Num_samples = 0;
    for i in range(0,len(signal)):
        Num_samples = Num_samples + signal[i].shape[0];
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr
#sigma
def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    return stdc;



def filtros(rutaWav):
    #se usara la que viene del filtro FIR , la cual se guar en un .wav como: senal_for_wav.wav
    
    senal_en_wav, Fmuestreo = librosa.load(rutaWav)
    
   
    LL = int(np.floor(np.log2(senal_en_wav.shape[0])));

    coeff = pywt.wavedec( senal_en_wav, 'db6', level=LL );
#funcion para el wavelet de python,retornara cA y cD
    thr = thselect(coeff);
    coeff_t = wthresh(coeff,thr);

    x_rec = pywt.waverec( coeff_t, 'db6');

    x_rec = x_rec[0:senal_en_wav.shape[0]];
 

    x_filt = np.squeeze(senal_en_wav - x_rec);
  
    plt.figure(figsize=(15,6))
  
    #se sacan las graficas pertinentes para comparar la señal que viene del filtro FIR y la que se obtiene mediante el 
    #filtro wavelet
  
    plt.figure(figsize=(15,6))
    plt.subplot(2,1,1)
    plt.subplots_adjust( wspace=1, hspace=1)


    plt.title('Filtro FIR: Pasa banda')
    librosa.display.waveplot(senal_en_wav, Fmuestreo);
    plt.ylabel('Amplitude')
    plt.grid()
    plt.subplot(2,1,2)
    plt.title('Filtrado Wavelet')
    librosa.display.waveplot(x_rec, Fmuestreo)
    plt.ylabel('Amplitude')
    plt.grid()
    return x_rec,Fmuestreo
    


# In[12]:


senalallfilt,Fmuestreo=filtros('senalq_for_wav.wav')
senal_for_cic=librosa.output.write_wav('senalq_for_cic.wav', senalallfilt, Fmuestreo)


# ### Extracción de ciclos y anotaciones

# In[13]:


def punto5(rutaWav,rutaTxt):
    matrix = np.loadtxt(rutaTxt)
    senal, Fmuestreo = librosa.load(rutaWav)
    duracion=librosa.get_duration(senal, Fmuestreo)
    tiempo=np.arange(0,duracion,1/Fmuestreo);
    df=pd.DataFrame(matrix)
    #df.to_excel ('C:\Users\W\Downloads\taller_audio_8_mayo\export_dataframe.xlsx')
    senalesR=[];
    for j in matrix:
        maxpeta=j[1]
        minpeta=j[0]
        x=[]
        y=[]
        for i in range(0,len(tiempo)):
            if(tiempo[i]<=maxpeta and tiempo[i]>=minpeta ):
                x.append(tiempo[i])
                y.append(senal[i])
        senalesR.append([x,y])
    #se carga el archivo txt de tal forma que se permita manejar este como una matriz y asi hacer las iteraciones necesarias 
    #para reconocer crepitaciones y sibilancias(todos esto se logra mediante "np.loadtxt" .
    #Para reconocer el tiempo de la señal se encuentra el vector de tiempo con el inicio, el final y el periodo de muestreo de la señal
    #Posterior a esto se reconocen en el vector de tiempo el inicio y el final de los ciclos segun las columnas del txt y se agregan a la vez
    #estos valores a dos vectores
    #y estos a su vez a una variable que los reconocera como valores en x y y
    cont=0
    texto=""
    #Se clasifican por crepitancias y sibilancias en:
    #Enfermos con crepitancias
    #Enfermos con sibilancias
    #Enfermos con ambas patologias
    #Sanos
    for k in range(0,len(senalesR)):
        if (matrix[cont][3] == 0 and matrix[cont][2] == 0):
            texto=(" Paciente sano , no presenta crepitancias ni sibilancias")
        elif(matrix[cont][3]==1) and (matrix[cont][2]==0):
            texto=(" Paciente enfermo, con sibilancias")
        elif(matrix[cont][3]==0) and (matrix[cont][2]==1):
            texto=(" Paciente enfermo, con crepitaciones")
        elif(matrix[cont][3]==1) and (matrix[cont][2]==1):
            texto=(" Paciente enfermo, con crepitaciones y sibilancias")
        plt.figure(k)
        plt.plot(senalesR[k][0], senalesR[k][1])

        plt.title(" Ciclo respiratorio "+str(cont +1)+"\n"+texto)
        plt.ylabel("Amplitud")
        plt.xlabel("Tiempo (de cada ciclo)")
        cont=cont+1
    

    


# In[14]:


def indices(segnal):
    #indices que se definen en el documento, varios de estos se han utilizado en el laboratorio. 
    varianza=np.var(segnal)
    maxs= max(segnal)
    mins = min(segnal)
    rango = abs(maxs - mins)
    moving_average = np.convolve(segnal, np.ones(800),'valid')/800
    suma=max(moving_average)
    f,Pxx=signal.periodogram(segnal)
    prome= np.mean(Pxx)
    return varianza,maxs,mins,rango,suma,prome


# In[15]:


def punto7(rutaWav,rutaTxt):
    #esta función se basa en los requrimientos de extracción de ciclos  y anotaciones , la misma que se hizo en el punto 5 
    #Pero en esta se hace una iteracion final para crear un dataFrame con las keys que se quieren 
    senal,Fmuestreo=filtros(rutaWav)
    matrix = np.loadtxt(rutaTxt)
    duracion=librosa.get_duration(senal, Fmuestreo)
    tiempo=np.arange(0,duracion,1/Fmuestreo)
    senalesR=[]
    for j in matrix:
        maxpeta=j[1]
        minpeta=j[0]
        x=[]
        y=[]
        for i in range(0,len(tiempo)):
            if(tiempo[i]<=maxpeta and tiempo[i]>=minpeta ):
                x.append(tiempo[i])
                y.append(senal[i])
        senalesR.append([x,y])
    cont=0
    texto=""
    df=[]
    for i in senalesR:
        varianza,maxs,mins,rango,suma,prome = indices(i[0])
        df.append([rutaWav,matrix[cont][0],matrix[cont][1],matrix[cont][2],matrix[cont][3],varianza,maxs,mins,rango,suma,prome])
        cont=cont+1
    dframe=pd.DataFrame(data=df,columns=["archivo","start","end","crepitancias","sibilancias","varianza","max","min","rango","suma","prom"])
    return dframe  


# In[16]:


def start(folder,range):
    contenido = os.listdir(folder)
    nombres=[]
    for i in contenido:
        nombre,ext=i.split(".")
        if(nombre not in nombres):
            nombres.append(nombre)
    dataframes=[]
    cont=0
    for i in nombres[range[0]:range[1]]:
        print(cont)
        dfi=punto7(folder+i+".wav",folder+i+".txt")
        dataframes.append(dfi)
        cont=cont+1
    return dataframes
#se realiza la iteracion para todos los archivos y se invoca la función " Punto 7"
        


# In[17]:


def concat(name,frames):
    framesF=pd.concat(frames)
    framesF.to_csv(name, encoding='utf-8', index=False)
  #permitira la concatenación de los data frames por lo que se hizo por rangos , para mejorar la velocidad del equipo  


# In[19]:


punto5('senalq_for_cic.wav','226_1b1_Pl_sc_LittC2SE.txt')
#se invoca la función punto 5 para un archivo en especifico
#se trabaja con la señal que viene de los ciclos


# In[20]:


punto7('senalq_for_cic.wav','226_1b1_Pl_sc_LittC2SE.txt')
#se realiza un ejemplo con una sola señal para verificar su funcionamiento 


# In[21]:


dataframes0100=start('./audio_and_txt_files/',[0,100]) 
#se realizan los DataFrame Para la ruta de archivos en un rango de 0-100


# In[22]:


concat('dataframes0100.csv',dataframes0100)
#seconcatenan los dataframe en este rango 


# In[23]:


dataframes100200=start('./audio_and_txt_files/',[100,200])
concat('dataframes100200.csv',dataframes100200)
#se realizan los DataFrame Para la ruta de archivos en un rango de 100-200
#seconcatenan los dataframe en este rango 


# In[24]:


dataframes200300=start('./audio_and_txt_files/',[200,300])
concat('dataframes200300',dataframes200300)
#se realizan los DataFrame Para la ruta de archivos en un rango de 200-300
#seconcatenan los dataframe en este rango 


# In[25]:


dataframes300400=start('./audio_and_txt_files/',[300,400])
concat('dataframes300400',dataframes300400)
#se realizan los DataFrame Para la ruta de archivos en un rango de 300-400
#seconcatenan los dataframe en este rango 


# In[26]:


dataframes400500=start('./audio_and_txt_files/',[400,500])
concat('dataframes400500',dataframes400500)
#se realizan los DataFrame Para la ruta de archivos en un rango de 400-500
#seconcatenan los dataframe en este rango 


# In[33]:


dataframes500600=start('./audio_and_txt_files/',[500,600])
concat('dataframes500600',dataframes500600)
#se realizan los DataFrame Para la ruta de archivos en un rango de 500-600
#seconcatenan los dataframe en este rango 


# In[27]:


dataframes600700=start('./audio_and_txt_files/',[600,700])
concat('dataframes600700',dataframes600700)
#se realizan los DataFrame Para la ruta de archivos en un rango de 600-700
#seconcatenan los dataframe en este rango 


# In[28]:


dataframes700800=start('./audio_and_txt_files/',[700,800])
concat('dataframes700800',dataframes700800)
#se realizan los DataFrame Para la ruta de archivos en un rango de 700-800
#seconcatenan los dataframe en este rango 


# In[29]:


dataframes800900=start('./audio_and_txt_files/',[800,900])
concat('dataframes800900',dataframes800900)
#se realizan los DataFrame Para la ruta de archivos en un rango de 800-900
#seconcatenan los dataframe en este rango 


# In[31]:


dataframes900920=start('./audio_and_txt_files/',[900,920])
concat('dataframes900920',dataframes900920)
#se realizan los DataFrame Para la ruta de archivos en un rango de 900-920
#seconcatenan los dataframe en este rango 


# In[34]:


df0100=pd.concat(dataframes0100)
df100200=pd.concat(dataframes100200)
df200300=pd.concat(dataframes200300)
df300400=pd.concat(dataframes300400)
df400500=pd.concat(dataframes400500)
df500600=pd.concat(dataframes500600)
df600700=pd.concat(dataframes600700)
df700800=pd.concat(dataframes700800)
df800900=pd.concat(dataframes800900)
df900920=pd.concat(dataframes900920)
#se definen cada uno de los dataframe como variables para concatenarlos

alldataframes=pd.concat([df0100,df100200,df200300,df300400,df400500,df500600,df600700,df700800,df800900,df900920])
#se concatenan todos


# In[35]:


df = pd.DataFrame(alldataframes)


# In[36]:


df.to_csv('alldataframess.csv')
#se guarda en un archo csv para un posterior analisis estadistico


# In[ ]:




