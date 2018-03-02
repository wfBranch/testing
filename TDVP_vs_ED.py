# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:57:29 2018

@author: Daniel
"""

import sys
# sys.path.insert(0, 'C:\Users\Daniel\Documents\Physics\Branching simulation\evoMPS package\evoMPS')#
import evoMPS.mps_gen as mp
import scipy as sp
import scipy.linalg as la
import numpy as np
import mps_ext as me
import evoMPS.tdvp_gen as tdvp
import functools as ft
import matplotlib.pyplot as plt

N=8
D=10
q=2

zero_tol = 1E-10  #Zero-tolerance for the Schmidt coefficients squared (right canonical form)
J=0.1

hamTerm = np.kron(me.Sx,me.Sx)+J*np.kron(me.Sz,np.eye(2)) #TFIM
hamTerm = me.random_hermitian(q**2)
hamTermList = me.make_ham_for_MPS(N,hamTerm)
hamFull=me.make_ham(hamTerm,N)
s = tdvp.EvoMPS_TDVP_Generic(N, np.ones(N+1,dtype=int)*D, np.ones(N+1,dtype=int)*q,hamTermList)

s.randomize()
s.zero_tol=zero_tol

for i in range(100):
    s.take_step_RK4(0.08)
    s.update()


plt.close('all')
me.plot_energy(s,'E gs')
s.apply_op_1s(me.Sz,3)
me.plot_energy(s,'E gs pert')


dt=0.01
nSteps=1000
T=nSteps*dt
tList=np.linspace(0,T,nSteps+1)
vListED=np.zeros((nSteps+1,2**N),dtype='complex') # ED: exact diagonalization
vListTN=np.zeros((nSteps+1,2**N),dtype='complex') # TN: tensor network
angleList=np.zeros(nSteps+1)
enListTN=np.zeros(nSteps+1)
enListED=np.zeros(nSteps+1)
vListTN[0,:]=me.vectorize(s).transpose()
enListTN[0]=s.H_expect.real
v0=me.vectorize(s)
enListED[0]=v0.transpose().conj().dot(hamFull.dot(v0)).real[0,0]
vListED[0,:]=v0.transpose()
# dU=la.expm(hamFull*1j*dt) # Why doesn't this work for matrix type? Bug in la.expm() for matrix type!
dU =la.expm(np.array(hamFull)*-1.j*dt)
for i in range(1,nSteps+1):
#    s.take_step(dt)
    s.take_step_RK4(dt*1.j)
#    s.update()
    v=me.vectorize(s)
    vListTN[i,:]=v.transpose()
    vListED[i,:]=dU.dot(vListED[i-1,:])
    angleList[i]=me.angle(vListTN[i,:],vListED[i,:])
#    angleList[i]=angle(me.partial_trace(vListTN[i,:],N,[3,4],complement=True),me.partial_trace(vListED[i,:],N,[3,4],complement=True))
    enListTN[i]=s.H_expect.real
    enListED[i]=v0.transpose().conj().dot(hamFull.dot(v0)).real[0,0]
    if i % 400 == 0:
        me.plot_energy(s,'E, t='+str(tList[i]))


me.plot_energy(s,'E final')

print('Angle final vs. initial:',angle(vListED[0,:],vListED[-1,:]))
print('Energy drift:',(np.abs(enListTN.max())-np.abs(enListTN.min())))
print('ED vs. TN energy:', la.norm(enListTN-enListED))
print('Final angle:',angleList[-1])
plt.figure()
plt.plot(tList,angleList)
plt.title('Angle')
plt.show()
