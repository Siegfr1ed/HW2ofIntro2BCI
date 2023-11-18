#!/usr/bin/env python
# coding: utf-8

import h5py
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
from scipy.stats import entropy, skew
from scipy.linalg import toeplitz, cholesky
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

mat_data = h5py.File('indy_20161005_06.mat','r')
mat_data_2 = h5py.File('indy_20160921_01.mat','r')

chan_names = mat_data.get('chan_names')
chan_names = np.array(chan_names)
chan_names = np.transpose(chan_names)
chan_names = chan_names.flatten()
cursor_pos = mat_data.get('cursor_pos')
cursor_pos = np.array(cursor_pos)
cursor_pos = np.transpose(cursor_pos)
finger_pos = mat_data.get('finger_pos')
finger_pos = np.array(finger_pos)
finger_pos =np.transpose(finger_pos)
spikes = mat_data['spikes']
spikes = mat_data[spikes[0,0]]
spikes = np.transpose(spikes)
t = mat_data.get('t')
t = np.array(t)
t = np.transpose(t)
t = t.flatten()
target_pos = mat_data.get('target_pos')
target_pos = np.array(target_pos)
target_pos = np.transpose(target_pos)
wf = mat_data['wf']
wf = mat_data[wf[0,0]]
wf = np.transpose(wf)

m,n=spikes.shape
result=[]
number=np.zeros(11)
for row in range(m):
    for col in range(n):
        spike_times=spikes
        if len(spike_times)>100:
            hist, bins = np.histogram(spike_times, bins=np.arange(min(spike_times),max(spike_times), 5))
            a1=abs(target_pos[:,0]-cursor_pos[:,0])
            a2=abs(target_pos[:,1]-cursor_pos[:,1])
            averages1 = []
            averages2 = []
            len_tt=int(len(a1)/(len(hist)-1))
            for i in range(0, len(a1), len_tt):
                avg1 = np.mean(a1[i:i+len_tt])
                avg2 = np.mean(a2[i:i+len_tt])
                averages1.append(avg1)
                averages2.append(avg2)
            X_combined = np.vstack((averages1, averages2)).T
            model = LinearRegression()
            model.fit(X_combined,hist)
            y_pred = model.predict(X_combined)
            r_squared = r2_score(hist, y_pred)
            result.append(r_squared)
            if r_squared>0.5:
                number[10]+=1
            else:
                number[int(r_squared/0.05)]+=1
number=number/sum(number)
#print(number)

chan_names = mat_data_2.get('chan_names')
chan_names = np.array(chan_names)
chan_names = np.transpose(chan_names)
chan_names = chan_names.flatten()
cursor_pos = mat_data_2.get('cursor_pos')
cursor_pos = np.array(cursor_pos)
cursor_pos = np.transpose(cursor_pos)
finger_pos = mat_data_2.get('finger_pos')
finger_pos = np.array(finger_pos)
finger_pos =np.transpose(finger_pos)
spikes = mat_data_2['spikes']
spikes = mat_data_2[spikes[0,1]]
spikes = np.transpose(spikes)
t = mat_data_2.get('t')
t = np.array(t)
t = np.transpose(t)
t = t.flatten()
target_pos = mat_data_2.get('target_pos')
target_pos = np.array(target_pos)
target_pos = np.transpose(target_pos)
wf = mat_data_2['wf']
wf = mat_data_2[wf[0,0]]
wf = np.transpose(wf)

m,n=spikes.shape
result=[]
number1=np.zeros(11)
for row in range(m):
    for col in range(n):
        spike_times=spikes
        if len(spike_times)>100:
            hist, bins = np.histogram(spike_times, bins=np.arange(min(spike_times),max(spike_times), 5))
            a1=abs(target_pos[:,0]-cursor_pos[:,0])
            a2=abs(target_pos[:,1]-cursor_pos[:,1])
            averages1 = []
            averages2 = []
            len_tt=int(len(a1)/(len(hist)-1))
            for i in range(0, len(a1), len_tt):
                avg1 = np.mean(a1[i:i+len_tt])
                avg2 = np.mean(a2[i:i+len_tt])
                averages1.append(avg1)
                averages2.append(avg2)
            X_combined = np.vstack((averages1, averages2)).T
            model = LinearRegression()
            model.fit(X_combined,hist)
            y_pred = model.predict(X_combined)
            r_squared = r2_score(hist, y_pred)
            result.append(r_squared)
            if r_squared>0.5:
                number1[10]+=1
            else:
                number1[int(r_squared/0.05)]+=1
number1=number1/sum(number1)
#print(number1)

bins =np.arange(0.025,0.575,0.05)
plt.bar(bins-0.005,number,width=0.01,edgecolor='k', alpha=0.5, label='20161005_06')
plt.bar(bins+0.005,number1,width=0.01,edgecolor='g', alpha=0.5, label='20160921_01')
plt.xlabel('R^2')
plt.ylabel('percentage')
plt.legend()
plt.show()