#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from pytmat import *
import matplotlib.pyplot as plt

wl = np.arange(0.3,900,1,dtype=np.float64)
L = len(wl)

n1 = np.ones(L, dtype=np.complex128)
n2 = 1.5*np.ones(L, dtype=np.complex128)
n3 = 2.5*np.ones(L, dtype=np.complex128)
n = np.array([n1,n3,n1],dtype=np.complex128)

d = np.array([500], dtype=np.float64)

phi = 0.0
theta = 0.0

new_data_te = DataPy(d,n,wl,theta,0.0)
new_data_tm = DataPy(d,n,wl,theta,90.0)

# print(new_data_te.get_phi())

result_te = new_data_te.simulate()
result_tm = new_data_tm.simulate()

# print(i.d,i.n,i.wl,i.theta,i.phi)
colors = ['red','black']
freq = 3e8/(wl*1e-6)*1e-12 # freq in THz
fig,ax = plt.subplots()
ax.plot(freq,result_te.t,label="T TE",color=colors[0])
ax.plot(freq,result_te.r,label="R TE",color=colors[0],linestyle='--')
ax.plot(freq,result_tm.t,label="T TM",color=colors[1])
ax.plot(freq,result_tm.r,label="R TM",color=colors[1],linestyle='--')
plt.legend()
ax.set_xlim([0.1,3])
plt.show()
# output_array = input_array.copy()