import MPS
import numpy as np
import scipy.linalg

n = 8
bond_dim = 2 * n
site_dim = 2

epsilon = 1

def random_unitary(n):
	
	random_matrix = np.zeros((n,n), dtype = np.complex_)
	random_matrix += np.random.normal(size = (n,n))
	random_matrix += 1j * np.random.normal(size = (n,n))
	u, p = scipy.linalg.polar(random_matrix)
	return u
	

for i in range(10):
	
	random_block_state = np.zeros((bond_dim, bond_dim, site_dim), dtype = np.complex_)
	
	q = 3
	
	random_block_state[:q, :q] += np.random.normal(size = (q, q, site_dim))
	random_block_state[:q, :q] += 1j * np.random.normal(size = (q, q, site_dim))
	random_block_state[q:,q:] += np.random.normal(size = (2*n - q, 2*n - q, site_dim))
	random_block_state[q:,q :] += 1j * np.random.normal(size = (2*n - q, 2*n - q, site_dim))
	
	random_block_state += np.random.normal(scale = epsilon/4, size = (2 * n,2 * n,site_dim))
	random_block_state += 1j * np.random.normal(scale = epsilon/4, size = (2 * n,2 * n,site_dim))
	
	P = random_unitary(bond_dim)
	Q = random_unitary(bond_dim)
	
	random_block_state = np.einsum("ab, bci, cd->adi", P, random_block_state, Q)
	
	print(MPS.num_blocks(random_block_state, epsilon))

print("\n")	

for i in range(10):
	

	random_state = np.zeros((bond_dim, bond_dim, site_dim), dtype = np.complex_)

	random_state += np.random.normal(size = (bond_dim, bond_dim, site_dim))
	random_state += 1j * np.random.normal(size = (bond_dim, bond_dim, site_dim))

	print(MPS.num_blocks(random_state, epsilon))
	
	
