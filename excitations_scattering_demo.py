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
#gsAlreadyFound = False
#exAlreadyFound = False
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


p1=0.9
p2 = -0.5
nev = 3 # number eigenvalues to find
exAlreadyFound=False
try:
    exAlreadyFound
except:
    exAlreadyFound = False
if not exAlreadyFound:
    ev1, eV1 = su.excite_top_triv(p1,nev=nev,return_eigenvectors=True)
    ev2, eV2 = su.excite_top_triv(p2,nev=nev,return_eigenvectors=True)
    exAlreadyFound=True
    

exct_ind1=1 # ind = 1 for first excitation
assert exct_ind1<nev
x1 = eV1[:,exct_ind1-1].reshape((D,q*D-D))
B1 = su.get_B_from_x(x1, su.Vsh[0], su.l_sqrt_i[0], su.r_sqrt_i[0])

exct_ind2=1 # ind = 1 for first excitation
assert exct_ind2<nev
x2 = eV2[:,exct_ind2-1].reshape((D,q*D-D))
B2 = su.get_B_from_x(x2, su.Vsh[0], su.l_sqrt_i[0], su.r_sqrt_i[0])

A = copy.copy(su.A[0])

N=60

hamTermList = me.make_ham_for_MPS(N,hamTerm.reshape(q**2,q**2))
s0 = me.make_trans_inv_mps_from_tensor(A,N)
s0 = me.convert_tdvp_mps(s0,hamTermList)
E0u = su.h_expect.real
E0 = s0.H_expect.real/N
print('E gs finite: '),E0
print('E gs uniform: '),E0u


mid = N//2-5
x0=N//4-5
DeltaX=5
s1 = me.make_wavepacket_from_tensor(A,B1,N,x0,DeltaX,p1,i1=1,i2=mid-2,do_update=False)
x0=3*N//4-10
DeltaX=5
s2 = me.make_wavepacket_from_tensor(A,B2,N,x0,DeltaX,p2,i1=mid+2,i2=N-1,do_update=False)
s = me.join_MPS(s1,s2,mid)
s = me.convert_tdvp_mps(s,hamTermList)
me.plot_energy(s,'initial wavepacket joined')
print('E excitation eigenvalues: '+str(ev1[exct_ind1-1]+ev1[exct_ind2-1]-2*E0u))
print('E initial: '+str(s.H_expect.real/N-E0))

calc_mi=False
zero_tol = 1E-10  #Zero-tolerance for the Schmidt coefficients squared (right canonical form)
#hamTerm = hamTerm.reshape(q**2,q**2)
#hamTermList = me.make_ham_for_MPS(N,hamTerm)

dyn_exp=False
dt=0.1
nSteps=2500
mi_trunc=5
if calc_mi:
    i1=int(round(N/4))+1
    i2=int(round(N/2))-1
    i3=int(round(N/2))+1
    i4=int(round(3*N/4))-1
    print('N,i1,i2,i3,i4: ' + str([N,i1,i2,i3,i4]))
    miList=np.zeros(nSteps+1)
    miList[0]=me.mutual_information_disconn(s,i1,i2,i3,i4,truncate=mi_trunc)

i5=mid

T=nSteps*dt
tList=np.linspace(0,T,nSteps+1)
sList=np.zeros(nSteps+1)
sList[0]=s.entropy(i5)

plot_energy_interval = 50 # how often to plot energy
mi_interval = 25 # how often to compute mutual information

print('Begin real time evolution')
for i in range(1,nSteps+1):
    if dyn_exp:
        dstep = dt**(5./2.)
        s.take_step(dstep * 1.j, dynexp=True, dD_max=4, sv_tol=1E-5)
    s.update()
    s.take_step_RK4(dt*1.j)
    if calc_mi:
        if i % mi_interval == 0:
            miList[i]=me.mutual_information_disconn(s,i1,i2,i3,i4,truncate=mi_trunc)
        else:
            miList[i]=miList[i-1]
    sList[i]=s.entropy(i5)
    if i % plot_energy_interval == 0:
        print('Real time evolution, progress=' + str(i*1.0/nSteps))
        me.plot_energy(s,'E, t='+str(i*dt))
        if calc_mi:
            print('Mutual information: %f'%miList[i])

me.plot_energy(s,'E final')

if calc_mi:
    plt.figure()
    plt.gcf().clear() 
    plt.plot(tList,miList)
    plt.title('MI')
    plt.figure()
plt.gcf().clear() 
plt.plot(tList,sList)
plt.title('S')
plt.show()

