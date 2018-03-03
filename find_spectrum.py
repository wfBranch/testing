import evoMPS.mps_gen as mp
import scipy as sp
import scipy.linalg as la
import numpy as np
import mps_ext as me
import evoMPS.tdvp_gen as tdvp
import functools as ft
import matplotlib.pyplot as plt
import evoMPS.tdvp_uniform as tdvpu
import copy

D=5
q=2

zero_tol=1E-10

nSteps = 100
step = 0.08

plt.close('all')
gsAlreadyFound = False
exAlreadyFound = False
try:
    gsAlreadyFound
except:
    gsAlreadyFound = False
if not gsAlreadyFound:
#    hamTerm = me.make_Heis_for_MPS(1,J, h)[1]
#    hamTerm = me.random_hermitian(q**2).reshape(q,q,q,q)
    J=0.1
    hamTerm = (J*np.kron(me.Sz,me.Sz)+np.kron(me.Sx,np.eye(2))).reshape(q,q,q,q) #TFIM
    su = tdvpu.EvoMPS_TDVP_Uniform(D, q,hamTerm)
    su.zero_tol = zero_tol
    su.randomize()
#    print 'Initial energy = ',su.h_expect.real
    print('Finding g.s.')
    for i in range(nSteps):
        su.take_step(step)
        su.update() 
    #    print 'Step %d'%i+', E=%f'%su.h_expect.real
#    print 'G.s. energy = ',su.h_expect.real
    gsAlreadyFound = True

pList = np.linspace(0,np.pi,num=50)
for i in range(len(pList)):
    ev1, eV1 = su.excite_top_triv(p1,nev=nev,return_eigenvectors=True) 

