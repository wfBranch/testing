import evoMPS.mps_gen as mp
import evoMPS.tdvp_gen as tdvp
import mps_ext as me
import numpy as np
import numpy.linalg as la
import copy

N=50
D=4
assert 2*D < 7 # Otherwise me.mutual_information_disconn() will take too long
q=2

ind = int(N/2)
s1 = mp.EvoMPS_MPS_Generic(N, np.ones(N+1,dtype=int)*D, np.ones(N+1,dtype=int)*q)
s1.randomize()
sch1 = s1.schmidt_sq(ind).real
print 'sch1: '+str(sch1)
s2 = mp.EvoMPS_MPS_Generic(N, np.ones(N+1,dtype=int)*D, np.ones(N+1,dtype=int)*q)
s2.randomize()
sch2 = s2.schmidt_sq(ind).real
print 'sch2: '+str(sch2)

s3 = me.add_MPS(s1,s2)
s3.update()
sch3 = s3.schmidt_sq(ind).real
print 'sch3: '+str(sch3)

sch3b = np.sort(np.concatenate((sch1/2,sch2/2)))
print 'sch3b: '+str(sch3b)
print '|sch3-sch3b|/sch3: '+str(np.abs(np.divide(sch3-sch3b,sch3)))
s4 = copy.deepcopy(s3)
s4.randomize()
sch4 = s4.schmidt_sq(ind).real

assert N>=20 # just for how I chose i2, i3 below
i1 = N/4
i2 = N/2-4
i3 = N/2+4
i4 = 3*N/4

if max(s3.D)**4<2**10+1:
    mi = me.mutual_information_disconn(s3,i1,i2,i3,i4)
    print 'mi: %f'%mi
    mi_r = me.mutual_information_disconn(s4,i1,i2,i3,i4)
    print 'mi rand: %f'%mi_r

assert ind<N and ind>0 # otherwise s.l[ind] might np.ndarray, not native matmult object
A = np.einsum('ijk,k -> ijk',s3.A[ind],1.0/s3.l[ind].diag)
X = np.abs(A[0,:,:])**2+np.abs(A[1,:,:])**2
np.set_printoptions(suppress=True)
print np.round(X)
