
import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from PIL import Image
import cv2
import pickle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import random
from numpy import asarray
from scipy.signal import stft
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
df_nose=pd.read_csv('/content/drive/MyDrive/fourier/nose_tran_400.csv',header=None)
df_et=pd.read_csv('/content/drive/MyDrive/fourier/ETdata_400.csv',header=None)
def plot_history(history, name):
    with plt.xkcd(scale=0.2):
      fig, ax = plt.subplots(1,2, figsize=(12,6))
      for i, metric in enumerate(['loss', 'accuracy']): 
          ax[i].plot(history.history[metric], label='Train', color='#EFAEA4',linewidth=3)
          ax[i].plot(history.history[f'val_{metric}'], label='Validation', color='#B2D7D0',linewidth=3)
          if metric == 'accuracy': 
            ax[i].axhline(0.5, color='#8d021f', ls='--', label='Trivial accuracy')
            ax[i].set_ylabel("Accuracy", fontsize=14)
          else:
            ax[i].set_ylabel("Loss", fontsize=14)
          ax[i].set_xlabel('Epoch', fontsize=14)

      plt.suptitle(f'{name} Training', y=1.05, fontsize=16)
      plt.legend(loc='best')
      plt.tight_layout()
df_comb=pd.concat([df_nose, df_et], join='inner', axis=1)

for i in range(400):
  comb=df_comb.iloc[i,:].to_numpy()
  nose=df_nose.iloc[i,:].to_numpy()
  tongue=df_et.iloc[i,:].to_numpy()
  
  comb_stan=(comb - np.mean(comb)) / np.std(comb) 
  nose_stan=(nose - np.mean(nose)) / np.std(nose) 
  tongue_stan=(tongue - np.mean(tongue)) / np.std(tongue) 


# Define the window length and overlap for the STFT
  window_length = 128
  overlap = 64

# Compute the STFT using the scipy library
  f, t, Zxx = stft(comb_stan, window='hamming', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'c_128_hamming.png'
  plt.savefig(sav)
  plt.clf()
  
  f, t, Zxx = stft(nose_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'n_128_hamming.png'
  plt.savefig(sav)
  plt.clf()

  
  f, t, Zxx = stft(tongue_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'t_128_hamming.png'
  plt.savefig(sav)
  plt.clf()

for i in range(400):
  comb=df_comb.iloc[i,:].to_numpy()
  nose=df_nose.iloc[i,:].to_numpy()
  tongue=df_et.iloc[i,:].to_numpy()
  
  comb_stan=(comb - np.mean(comb)) / np.std(comb) 
  nose_stan=(nose - np.mean(nose)) / np.std(nose) 
  tongue_stan=(tongue - np.mean(tongue)) / np.std(tongue) 


# Define the window length and overlap for the STFT
  window_length = 64
  overlap = 32

# Compute the STFT using the scipy library
  f, t, Zxx = stft(comb_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'c_64_hann.png'
  plt.savefig(sav)
  plt.clf()
  
  f, t, Zxx = stft(nose_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'n_64_hann.png'
  plt.savefig(sav)
  plt.clf()

  
  f, t, Zxx = stft(tongue_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'t_64_hann.png'
  plt.savefig(sav)
  plt.clf()

for i in range(400):
  comb=df_comb.iloc[i,:].to_numpy()
  nose=df_nose.iloc[i,:].to_numpy()
  tongue=df_et.iloc[i,:].to_numpy()
  
  comb_stan=(comb - np.mean(comb)) / np.std(comb) 
  nose_stan=(nose - np.mean(nose)) / np.std(nose) 
  tongue_stan=(tongue - np.mean(tongue)) / np.std(tongue) 


# Define the window length and overlap for the STFT
  window_length = 32
  overlap = 16

# Compute the STFT using the scipy library
  f, t, Zxx = stft(comb_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'c_32_hann.png'
  plt.savefig(sav)
  plt.clf()
  
  f, t, Zxx = stft(nose_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'n_32_hann.png'
  plt.savefig(sav)
  plt.clf()

  
  f, t, Zxx = stft(tongue_stan, window='hann', nperseg=window_length, noverlap=overlap)

# Plot the resulting spectrogram
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(np.abs(Zxx)))
  st=str(i)
  sav=st+'t_32_hann.png'
  plt.savefig(sav)
  plt.clf()

arr=np.array([1,0,0,0]*100)
arr2=np.array([0,1,0,0]*100)
arr3=np.array([0,0,1,0]*100)
arr4=np.array([0,0,0,1]*100)
y=np.vstack((arr,arr2,arr3,arr4)).reshape(400,4)

k=asarray(Image.open('0n_128_hamming.png'))
for i in range(1,400):
  st=str(i)+'n_128_hamming.png'
  img = Image.open(st)
  numpydata = asarray(img)
  k=np.vstack((k,numpydata))

x=k.reshape(400,480,640,4)
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

from tensorflow.keras import regularizers

model_nose = models.Sequential()
model_nose.add(layers.Conv2D(4, (2, 2), activation='relu', input_shape=(480,640,4)))#, kernel_regularizer=regularizers.L2(0.01), bias_regularizer=regularizers.L2(0.01)))
model_nose.add(layers.MaxPooling2D((2, 2)))
model_nose.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_nose.add(layers.MaxPooling2D((2, 2)))
model_nose.add(tf.keras.layers.BatchNormalization())
model_nose.add(layers.Conv2D(16, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_nose.add(layers.MaxPooling2D((2, 2)))
model_nose.add(tf.keras.layers.BatchNormalization())
model_nose.add(layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_nose.add(layers.Flatten())
model_nose.add(layers.Dense(4, activation='softmax', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_nose.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_nose.summary()

history_nose=model_nose.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=15, batch_size=10)

results_nose = model_nose.evaluate(X_test, y_test, batch_size=10)
print("test loss, test acc:", results_nose)

plot_history(history_nose, 'STFT images  of e_nose signals used for')

k=asarray(Image.open('0t_128_hamming.png'))
for i in range(1,400):
  st=str(i)+'t_128_hamming.png'
  img = Image.open(st)
  numpydata = asarray(img)
  k=np.vstack((k,numpydata))

x=k.reshape(400,480,640,4)
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

from tensorflow.keras import regularizers

model_ton = models.Sequential()
model_ton.add(layers.Conv2D(4, (2, 2), activation='relu', input_shape=(480,640,4)))#, kernel_regularizer=regularizers.L2(0.01), bias_regularizer=regularizers.L2(0.01)))
model_ton.add(layers.MaxPooling2D((2, 2)))
model_ton.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_ton.add(layers.MaxPooling2D((2, 2)))
model_ton.add(tf.keras.layers.BatchNormalization())
model_ton.add(layers.Conv2D(16, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_ton.add(layers.MaxPooling2D((2, 2)))
model_ton.add(tf.keras.layers.BatchNormalization())
model_ton.add(layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_ton.add(layers.Flatten())
model_ton.add(layers.Dense(4, activation='softmax', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_ton.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_ton.summary()

history_ton_com=model_ton.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=15, batch_size=10)

results_tongue = model_ton.evaluate(X_test, y_test, batch_size=10)
print("test loss, test acc:", results_tongue)

plot_history(history_ton_com, 'STFT images  of e_tongue signals  are used for')

k=asarray(Image.open('0c_128_hamming.png'))
for i in range(1,400):
  st=str(i)+'c_128_hamming.png'
  img = Image.open(st)
  numpydata = asarray(img)
  k=np.vstack((k,numpydata))

x=k.reshape(400,480,640,4)
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

from tensorflow.keras import regularizers

model_com = models.Sequential()
model_com.add(layers.Conv2D(4, (2, 2), activation='relu', input_shape=(480,640,4)))#, kernel_regularizer=regularizers.L2(0.01), bias_regularizer=regularizers.L2(0.01)))
model_com.add(layers.MaxPooling2D((2, 2)))
model_com.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_com.add(layers.MaxPooling2D((2, 2)))
model_com.add(tf.keras.layers.BatchNormalization())
model_com.add(layers.Conv2D(16, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_com.add(layers.MaxPooling2D((2, 2)))
model_com.add(tf.keras.layers.BatchNormalization())
model_com.add(layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_com.add(layers.Flatten())
model_com.add(layers.Dense(4, activation='softmax', kernel_regularizer=regularizers.L2(0.01)))#, bias_regularizer=regularizers.L2(0.02)))
model_com.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_com.summary()

history_com=model_com.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=15, batch_size=20)

plot_history(history_com, 'STFT images  of e_tongue and e_nose signals combined are used for')

class_A= df_et.iloc[0]
plt.rcParams['figure.figsize'] = [10, 4]
class_A.plot(kind='line')
fig=plt.xlabel('Measurments Points',fontweight='bold')
fig=plt.ylabel('Response',fontweight='bold')
fig=plt.title('Electronic Nose Signal', fontweight='bold')




#class_A=df_nose.iloc[0]
#class_B=df_nose.iloc[101]
#class_C=df_nose.iloc[201]
#class_D=df_nose.iloc[301]
#plt.rcParams['figure.figsize'] = [10, 4]
#class_A.plot(kind='line',color="red", label='Class A')
#class_B.plot(kind='line',color="blue", label='Class B')
#class_C.plot(kind='line',color="green", label='Class C')
#class_D.plot(kind='line',color="black", label='Class D')

#fig=plt.xlabel('Measurments Points',fontweight='bold')
#fig=plt.ylabel('Response',fontweight='bold')
#fig=plt.title('Different Classes of Tea- Electronic Tongue Signal', fontweight='bold')
#fig=plt.legend(loc='upper center')
savefig('(T)nose_signal.tiff')

min=0
num=0
for x in range(2500,3000):
  if df_nose.iloc[0][x]<min:
    num=x
    min=df_nose.iloc[0][x]

    
print(num)
    
df_nose.iloc[0][2777] #695,1460,2115#2777

E1=df_nose.iloc[0][0:695]
E2=df_nose.iloc[0][695:1460]
E3=df_nose.iloc[0][1460:2115]
E4=df_nose.iloc[0][2115:2777]
E5=df_nose.iloc[0][2777:]


plt.rcParams['figure.figsize'] = [10, 3]
E1.plot(kind='line',color="red",label='GOLD')
E2.plot(kind='line',color="green",label='IRRIDIUM')
E3.plot(kind='line',color="blue",label='PALLADIUM')
E4.plot(kind='line',color="yellow",label='PLATINUM')
E5.plot(kind='line',color="black",label='RHODIUM')

fig=plt.xlabel('Measurments Points',fontweight='bold')
fig=plt.ylabel('Response',fontweight='bold')
fig=plt.title('Different Types of Electrodes in Electronic Tongue', fontweight='bold')
fig=plt.legend(loc='upper center')
savefig('(T)tongue_electrodes.tiff')