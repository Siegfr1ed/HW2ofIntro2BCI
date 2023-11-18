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
from numpy.linalg import inv as inv

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
finger_pos = np.transpose(finger_pos)
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

bin_width=0.04
bin_num=int(bin_width/(t[1]-t[0]))
min_t=min(t)
max_t=max(t)
bins =np.arange(min_t,max_t+bin_width,bin_width)
pos_bins=np.array([cursor_pos[k] for k in range(0,len(cursor_pos),10)])

k=0;
m,n=spikes.shape
spikes_data=[]
for row in range(m):
    tmp=[]
    for col in range(n):
        tmp.append(spikes[row,col])
    if len(tmp)>0:
        k+=1
        spikes_data.append(tmp)

neural_data=np.zeros((len(pos_bins)-1,k))

for key,value in enumerate(spikes_data):
    hist, _ = np.histogram(value, bins)
    neural_data[:,key]=hist.T

X_kf=neural_data

speed=np.zeros(pos_bins.shape)
time_bin=t[1]-t[0]
for i in range(1,pos_bins.shape[0]): 
    speed[i,0]=(pos_bins[i,0]-pos_bins[i-1,0])/bin_width
    speed[i,1]=(pos_bins[i,1]-pos_bins[i-1,1])/bin_width
temp=np.diff(speed,axis=0)

acc_binned=np.concatenate((temp[:1,:],temp),axis=0)

print(speed.shape,acc_binned.shape,pos_bins.shape)
y_kf=np.concatenate((pos_bins,speed,acc_binned),axis=1)

print(y_kf.shape)

def get_R2(y_test,y_test_pred):

    R2_list=[]
    for i in range(y_test.shape[1]):
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2)
    R2_array=np.array(R2_list)
    return R2_array

def get_rho(y_test,y_test_pred):
    rho_list=[]
    for i in range(y_test.shape[1]):
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho)
    rho_array=np.array(rho_list)
    return rho_array

class KalmanFilterDecoder(object):

    def __init__(self,C=1):
        self.C=C


    def fit(self,X_kf_train,y_train):

        X=np.matrix(y_train.T)
        Z=np.matrix(X_kf_train.T)
        nt=X.shape[1]
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T)
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C
        H = Z*X.T*(inv(X*X.T))
        Q = ((Z - H*X)*((Z - H*X).T)) / nt
        params=[A,W,H,Q]
        self.model=params

    def predict(self,X_kf_test,y_test):
        A,W,H,Q=self.model
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)
        num_states=X.shape[0]
        states=np.empty(X.shape)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0]
        states[:,0]=np.copy(np.squeeze(state))
        for t in range(X.shape[1]-1):
            P_m=A*P*A.T+W
            state_m=A*state
            K=P_m*H.T*inv(H*P_m*H.T+Q)
            P=(np.matrix(np.eye(num_states))-K*H)*P_m
            state=state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1]=np.squeeze(state)
        y_test_predicted=states.T
        return y_test_predicted
training_range=[0, 0.9]
valid_range=[0.9,1]

num_examples_kf=X_kf.shape[0]
training_set=np.arange(int(np.round(training_range[0]*num_examples_kf))+1,int(np.round(training_range[1]*num_examples_kf))-1)
valid_set=np.arange(int(np.round(valid_range[0]*num_examples_kf))+1,int(np.round(valid_range[1]*num_examples_kf))-1)

print(training_set)
X_kf_train=X_kf[training_set,:]
y_kf_train=y_kf[training_set,:]
X_kf_valid=X_kf[valid_set,:]
y_kf_valid=y_kf[valid_set,:]
X_kf_train_mean=np.nanmean(X_kf_train,axis=0)
X_kf_train_std=np.nanstd(X_kf_train,axis=0)

X_kf_train=(X_kf_train-X_kf_train_mean)/X_kf_train_std
X_kf_valid=(X_kf_valid-X_kf_train_mean)/X_kf_train_std
y_kf_train_mean=np.mean(y_kf_train,axis=0)
y_kf_train=y_kf_train-y_kf_train_mean
y_kf_valid=y_kf_valid-y_kf_train_mean

print("同时使用位置/速度/加速度")

predict_num=3
model_kf=KalmanFilterDecoder(C=1)

model_kf.fit(X_kf_train,y_kf_train)

y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid)

R2_kf=get_R2(y_kf_valid,y_valid_predicted_kf)
print('R2:',R2_kf)
rho_kf=get_rho(y_kf_valid,y_valid_predicted_kf)
print('rho2:',rho_kf**2)

fig_x_kf=plt.figure()
plt.plot(y_kf_valid[:,predict_num]+y_kf_train_mean[predict_num-1],color='b',label="real")
plt.plot(y_valid_predicted_kf[:,predict_num]+y_kf_train_mean[predict_num-1],color='r',label="predict")
plt.legend()
plt.show()


print("使用位置/速度")

kk=4
predict_num=3
model_kf=KalmanFilterDecoder(C=1)
model_kf.fit(X_kf_train,y_kf_train[:,:kk])

y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid[:,:kk])

print(y_kf_train[:,:kk].shape,y_kf_train.shape,y_valid_predicted_kf.shape)

R2_kf=get_R2(y_kf_valid[:,:kk],y_valid_predicted_kf)
print('R2:',R2_kf)
rho_kf=get_rho(y_kf_valid[:,:kk],y_valid_predicted_kf)
print('rho2:',rho_kf**2)

fig_x_kf=plt.figure()
plt.plot(y_kf_valid[:,predict_num]+y_kf_train_mean[predict_num-1],color='b',label="real")
plt.plot(y_valid_predicted_kf[:,predict_num]+y_kf_train_mean[predict_num-1],color='r',label="predict")
plt.legend()
plt.show()

print("单独使用位置/速度/加速度")

predict_num=1
for kk in range(2,7,2):
    model_kf=KalmanFilterDecoder(C=1)
    model_kf.fit(X_kf_train,y_kf_train[:,kk-2:kk])
    y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid[:,kk-2:kk])
    print(y_kf_train[:,kk-2:kk].shape,y_valid_predicted_kf.shape)
    R2_kf=get_R2(y_kf_valid[:,kk-2:kk],y_valid_predicted_kf)
    print('R2:',R2_kf)
    rho_kf=get_rho(y_kf_valid[:,kk-2:kk],y_valid_predicted_kf)
    print('rho2:',rho_kf**2)

    fig_x_kf=plt.figure()
    plt.plot(y_kf_valid[:,kk-2]+y_kf_train_mean[kk-2],color='b',label="real")
    plt.plot(y_valid_predicted_kf[:,0]+y_kf_train_mean[kk-2],color='r',label="predict")
    plt.legend()
    plt.show()

from sklearn.linear_model import LinearRegression
predict_num=2
model = LinearRegression()

model_kf.fit(X_kf_train,y_kf_train)

y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid)

R2_kf=get_R2(y_kf_valid,y_valid_predicted_kf)
print('R2:',R2_kf)
rho_kf=get_rho(y_kf_valid,y_valid_predicted_kf)
print('rho2:',rho_kf**2)


fig_x_kf=plt.figure()
plt.plot(y_kf_valid[:,predict_num]+y_kf_train_mean[predict_num-1],color='b',label="real")
plt.plot(y_valid_predicted_kf[:,predict_num]+y_kf_train_mean[predict_num-1],color='r',label="predict")
plt.legend()
plt.show()

class LSTMRegression(object):

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        model=Sequential()
        if keras_v1:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout))
        else:
            model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout))
        if self.dropout!=0: model.add(Dropout(self.dropout))
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
        if keras_v1:
            model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose)
        else:
            model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose)
        self.model=model


    def predict(self,X_test):
        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted

predict_num=2
model = LSTMRegression()

model_kf.fit(X_kf_train,y_kf_train)

y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid)

R2_kf=get_R2(y_kf_valid,y_valid_predicted_kf)
print('R2:',R2_kf)
rho_kf=get_rho(y_kf_valid,y_valid_predicted_kf)
print('rho2:',rho_kf**2)


fig_x_kf=plt.figure()
plt.plot(y_kf_valid[:,predict_num]+y_kf_train_mean[predict_num-1],color='b',label="real")
plt.plot(y_valid_predicted_kf[:,predict_num]+y_kf_train_mean[predict_num-1],color='r',label="predict")
plt.legend()
plt.show()