#!/usr/bin/env python
# coding: utf-8

import h5py
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
from scipy.stats import entropy, skew
from scipy.linalg import toeplitz, cholesky

mat_data = h5py.File('indy_20161005_06.mat','r')

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

time = np.linspace(0, 1, 64)
voltage = wf[0,:]
plt.figure(figsize=(10, 5))
plt.plot(time, voltage, color='b', label='Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.title('Neuron Voltage Waveform')
plt.legend()
plt.grid()
plt.show()

spike_times=spikes.T
plt.figure(figsize=(10, 5))
sample=1404
plt.eventplot(spike_times[0].flatten()[0:sample], lineoffsets=0.5, colors='b', label='Spike u0')
plt.xlabel('Time (ms)')
plt.title('Raster Plot')
plt.legend()
plt.grid()
plt.show()

psth_data=spike_times[0].flatten()
#print(psth_data.shape)
bin_width = 5
bins = np.arange(min(psth_data),max(psth_data), bin_width)
hist, _ = np.histogram(psth_data, bins)
#print(hist.shape,min(psth_data),max(psth_data))
hist = hist / bin_width
plt.figure(figsize=(10, 5))
plt.bar(bins[:-1], hist, width=bin_width, align='edge', color='b', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (spikes/s)')
plt.title('PSTH (Peri-Stimulus Time Histogram)')
plt.grid()
plt.show()

def get_gap(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

t_min=min(t)
t_max=max(t)
time_data=spike_times[0].flatten()
time_data = time_data[(time_data>= t_min) & (time_data <= t_max)]
bin_width = 1
bins = np.arange(t_min,t_max+bin_width, bin_width)
hist, _ = np.histogram(time_data, bins)
index=np.array([key for key,value in enumerate(t) if abs(value-int(value))<1e-6 and key!=0])
pos_gap=[get_gap(target_pos[i],cursor_pos[i]) for i in index]
stacked_array = np.stack((pos_gap, hist), axis=1)
sorted_indices = np.argsort(stacked_array[:, 0])
sorted_data = stacked_array[sorted_indices]
rate_data=np.zeros(150)
num=np.zeros(150)
for v in sorted_data:
    rate_data[int(v[0])]+=v[1]

plt.scatter(range(len(rate_data)), rate_data, marker='o', color='b')
plt.xlabel('Position Gap')
plt.ylabel('Firing number (spikes)')
plt.title('tuning curve')
plt.grid()
plt.show()