import numpy as np
import pdb
from scipy import linalg
import copy


#Apply two-site gate to gamma1 and gamma2.
#Input parameters: gamma1, gamma2 matrices at two sites
#Lambda 
#Alpha, eta 

#Values smaller than this are deleted(perhaps remove this...)
epsilon = 1e-10

#(...(ab)...) --> (...ab...) a and b are now different axes, though adjacent to each other
def split_axes(M, i, m, n):
	
	assert M.shape[i] == m * n
	
	M_spl = np.array( np.split(M, n, axis = i) )
	M_spl = np.moveaxis(M_spl, 0, i + 1)
	
	return M_spl
	
#(...a...b...) -->  (...(ab)...) Index ab in place where a used to be
def merge_axes(M, i, j):
	
	M_mer = np.moveaxis(M, j, 0)
	if i > j:
		M_mer = np.concatenate(M_mer, axis = i - 1)
	else:
		M_mer = np.concatenate(M_mer, axis = i)
		
	return M_mer
	
def matr_tensor(A, B):
	AB = np.einsum('ij,kl->ikjl', A, B)
	AB = merge_axes(AB, 0, 1)
	AB = merge_axes(AB, 1, 2)
	return AB


def apply_double_gate(U, gamma1, lmbda, gamma2, alpha, eta):
	
	bond_dim = lmbda.shape[0]
	site_dim = gamma1.shape[2]

	#In fact alpha and eta should be real, but just in case...
	alpha_conj = np.conj(alpha)
	eta_conj = np.conj(eta)
	
	eta_sq = eta * eta_conj
	alpha_sq = alpha * alpha_conj
	
	#Apply gate
	Theta = np.einsum('klij, abk, b, byl->ayij', U, gamma1, lmbda, gamma2, dtype = np.complex_)
	
	Theta_conj = np.conj(Theta)	
	
	#Matrix is converted to form mutliplied by etas at this point, must be de-converted later
	rho_DK = np.einsum('a,ayij,aYiJ,y,Y->yjYJ', alpha_sq, Theta, Theta_conj, eta, eta_conj,  dtype = np.complex_)
	
	rho_DK_sq = merge_axes(rho_DK, 0, 1)
	rho_DK_sq = merge_axes(rho_DK_sq, 1, 2)
	
	rho_DK_eigenval, rho_DK_eigenvec = linalg.eig(rho_DK_sq)
	
	rho_DK_eigenvec = [ rho_DK_eigenvec[:,k] for k in range(len(rho_DK_eigenvec))]
	
	
	rho_DK_reconstructed = np.zeros(rho_DK_sq.shape, dtype = np.complex_)
	for k in range(len(rho_DK_eigenval)):
		eigenvec_matr = np.einsum('a,b->ab', rho_DK_eigenvec[k], np.conj(rho_DK_eigenvec[k]))
		rho_DK_reconstructed += rho_DK_eigenval[k] * eigenvec_matr
	
	
	eigenlist = [ (rho_DK_eigenval[k], rho_DK_eigenvec[k]) for k in range(len(rho_DK_eigenval))]
	
	eigenlist.sort(key = lambda x: -np.abs(x[0]))
	
	
	new_gamma1 = np.zeros(gamma1.shape, dtype = np.complex_)
	new_gamma2 = np.zeros(gamma2.shape, dtype = np.complex_)
	new_lmbda = np.zeros(lmbda.shape, dtype = np.complex_)
	
	#obtain updated gamma2
	for k in range(bond_dim):
		vec =  split_axes(eigenlist[k][1], 0, bond_dim, site_dim)
		new_gamma2[k] = vec
	
	#Set tiny values to zero(I think these are mostly numerical inaccuracies...?...need to think about this more...)
	for x in np.nditer(new_gamma2, op_flags = ['readwrite']):
		if np.abs(x) < epsilon:
			x[...] = 0
	
	
	for k in range(bond_dim):
		if np.abs(eta[k]) > 1e-12:
			new_gamma2[:,k] = new_gamma2[:,k] / eta[k]
		else:
			new_gamma2[:,k] = np.zeros(new_gamma2[:,k].shape)
	

	for k in range(bond_dim):
		
		row = np.einsum('yj,ayij,y->ai', np.conj(new_gamma2)[k], Theta, eta_sq, dtype = np.complex)
		
		#row = new_gamma1[:,k,:]
		row_length = np.abs(np.sqrt(np.einsum('aj,aj,a->',row, np.conj(row), alpha_sq)))
		
		new_lmbda[k] = row_length
		
		if new_lmbda[k] < 1e-12:
			new_lmbda[k] = 0
			new_gamma2[k] = np.zeros(new_gamma2[k].shape)
			new_gamma1[:,k,:] = np.zeros(new_gamma1[:,k,:].shape)
		else:
			new_gamma1[:,k,:] = row / new_lmbda[k]
	
	for x in np.nditer(new_gamma1, op_flags = ['readwrite']):
		if np.abs(x) < epsilon:
			x[...] = 0
	
	#Normalize
	new_gamma1 = new_gamma1 * np.sqrt(np.sum(new_lmbda**2))
	new_lmbda = new_lmbda / np.sqrt(np.sum(new_lmbda**2))
	
	return new_gamma1, new_lmbda, new_gamma2


def apply_single_gate(U,gamma):
	return  np.einsum('abi,ij->abj', gamma,U)
	
#Local measurement at gamma
def measurement(M, gamma, alpha, eta):
	
	alpha_conj = np.conj(alpha)
	alpha_sq = alpha * alpha_conj
	
	eta_conj = np.conj(eta)
	eta_sq = eta * eta_conj
	
	expec = np.einsum('ayi,ayj,a,y,ij->', np.conj(gamma), gamma, alpha_sq, eta_sq, M)
	return np.abs(expec) #This should be real anyway
	

#Local projection at a site, and normalization. Note that this will 
#mess up the left and right Schmidt decompositions, and these will have to be
#updated.
def project(P, alpha, gamma, eta):
	
	alpha_sq = alpha * np.conj(alpha)
	eta_sq = eta * np.conj(eta)
	
	new_gamma = np.einsum('ij,ayj->ayi', P, gamma)
	
	
	new_gamma_norm = np.einsum('ayi,ayi,a,y->', new_gamma, np.conj(new_gamma), alpha_sq, eta_sq)
		
	if new_gamma_norm < 1e-12:
		new_gamma = np.zeros(new_gamma.shape)
	else:	
		new_gamma = new_gamma / np.sqrt(new_gamma_norm)

	
	return new_gamma
	
#Updates the Schimidt decomposition. 
#That is, we will 'fix' gamma2's left Schmidt decomp. 
def update_schmidt_left(gamma1, alpha, gamma2, eta):
	
	bond_dim = gamma1.shape[0]
	site_dim = gamma1.shape[2]
	
	#Incorporate the lmbdas into the entries of gamma1
	gamma_actual = np.einsum('a,y,ayi->ayi', alpha, eta, gamma2)
	#Now we can treat alpha and eta as orthnormal basis for left and right of our site
	
	gamma_matrix = merge_axes(gamma_actual, 1, 2)
	
	U, S, V = np.linalg.svd(gamma_matrix)
	
	new_gamma1 = np.zeros(gamma1.shape, dtype = np.complex_)
	new_alpha = np.zeros(eta.shape, dtype = np.complex_)
	new_gamma2 = np.zeros(gamma2.shape, dtype = np.complex_)
	
	new_alpha = S[:bond_dim]
	
	for k in range(bond_dim):
		new_gamma2[k] = split_axes(V[k], 0, bond_dim, site_dim)
	#I don't think that I need to 're-incorporate' size of new_alpha's, 
	#I believe that the svd does that for me
	
	#Re-incorporate size of eta's
	for k in range(bond_dim):
		if np.abs(eta[k]) < 1e-12:
			new_gamma2[:,k] = np.zeros(new_gamma2[:,k].shape)
		else:
			new_gamma2[:,k] = new_gamma2[:,k] / eta[k]
	
	#V interpreted as matrix of new basis coefficients for gamma2 
	new_gamma1 = np.einsum('ik,aij->akj', U, gamma1)
	
	#kill off rows/columns attached to tiny lambda values
	#These shouldn't be a problem anyway, but they annoy me and 
	#seems like they could mess up things in some subtle way later
	for k in range(bond_dim):
		if np.abs(new_alpha[k]) < 1e-12:
			new_alpha[k] = 0
			new_gamma2[k] = np.zeros(new_gamma2[k].shape)
			new_gamma1[:,k] = np.zeros(new_gamma1[:,k].shape)
	
	return new_gamma1, new_alpha, new_gamma2

#Fixes gamma1's right Schmidt decomposition
def update_schmidt_right(alpha, gamma1, eta, gamma2):
	
	bond_dim = gamma1.shape[0]
	site_dim = gamma1.shape[2]
	
	#Incorporate the lmbdas into the entries of gamma1
	gamma_actual = np.einsum('a,y,ayi->ayi', alpha, eta, gamma1)
	#Now we can treat alpha and eta as orthnormal basis for left and right of our site
	
	gamma_matrix = merge_axes(gamma_actual, 0, 2)
	
	U, S, V = np.linalg.svd(gamma_matrix)
	
	new_gamma1 = np.zeros(gamma1.shape, dtype = np.complex_)
	new_eta = np.zeros(eta.shape, dtype = np.complex_)
	new_gamma2 = np.zeros(gamma2.shape, dtype = np.complex_)
	
	new_eta = S[:bond_dim]
	
	for k in range(bond_dim):
		new_gamma1[:,k] = split_axes(U[:,k], 0, bond_dim, site_dim)
	#I don't think that I need to 're-incorporate' size of new_eta's, 
	#I believe that the svd does that for me
	
	#Re-incorporate size of alpha's
	for k in range(bond_dim):
		if np.abs(alpha[k]) < 1e-12:
			new_gamma1[k] = np.zeros(new_gamma1[k].shape)
		else:
			new_gamma1[k] = new_gamma1[k] / alpha[k]
	
	#V interpreted as matrix of new basis coefficients for gamma2 
	new_gamma2 = np.einsum('ki,iyj->kyj', V, gamma2)
	
	#kill off rows/columns attached to tiny lambda values
	#These shouldn't be a problem anyway, but they annoy me and 
	#seems like they could mess up things in some subtle way later
	for k in range(bond_dim):
		if np.abs(new_eta[k]) < 1e-12:
			new_eta[k] = 0
			new_gamma1[:,k] = np.zeros(new_gamma1[:,k].shape)
			new_gamma2[k] = np.zeros(new_gamma2[k].shape)
	
	return new_gamma1, new_eta, new_gamma2
	
#Locally projects at site i using projector P, then updates the Schmidt decompositions
#to the left and right.
def project_and_update(gamma_list, lmbda_list, P, i):

	L = len(gamma_list)
	
	gamma_list[i] = project(P, lmbda_list[i - 1], gamma_list[i], lmbda_list[i])
	
	#Update left Schmidt decompositions
	
	for k in range(i - 1, -1, -1): 
		gamma_list[k], lmbda_list[k], gamma_list[k + 1] = \
			update_schmidt_left(gamma_list[k], lmbda_list[k], gamma_list[k + 1], lmbda_list[k + 1])
		
	#Right schmidt decompositions
	for k in range(i, L - 1):
		gamma_list[k], lmbda_list[k], gamma_list[k + 1] = \
			update_schmidt_right( lmbda_list[k - 1], gamma_list[k], lmbda_list[k], gamma_list[k + 1])
	
	
	return gamma_list, lmbda_list
	
#############################################
############### BETA FORMAT #################
#############################################

#Convert to beta form, useful for iterating applying double gates.
def convert_to_beta(gamma_list, lmbda_list):
	
	beta_list = [ np.einsum('abi,b->abi', gamma_list[k], lmbda_list[k]) for k in range(len(gamma_list)) ]
	
	return beta_list, lmbda_list
	
def convert_from_beta(beta_list, lmbda_list):
	
	bond_dim = beta_list[0].shape[0]
	
	gamma = [np.zeros(beta_list[i].shape, dtype = np.complex_) for i in range(len(beta_list))]
	lmbda = [np.zeros(lmbda_list[i].shape, dtype = np.complex_) for i in range(len(beta_list))]
	
	for i in range(len(beta_list)):
		for k in range(bond_dim):
			if np.abs(lmbda_list[i][k]) < 1e-12:
				#leave gamma_list[i][:,k] as zeros
				lmbda[i][k] = 0
			else:
				lmbda[i][k] = lmbda_list[i][k]
				gamma[i][:,k] = beta_list[i][:,k] / lmbda[i][k]
	
	return gamma, lmbda


#Apply the gate U to sites at beta1 and beta2. Iterating this transformation leads
#to greater numerical stability because there is no division by small singular values
def apply_double_gate_beta(U, alpha, beta1, beta2):
	
	bond_dim = beta1.shape[0]
	site_dim = beta1.shape[2]
	
	Psi = np.einsum('abi,byj->ayij', beta1, beta2)
	
	Theta = np.einsum('ayij,ijkl->aykl', Psi, U)
	
	Theta_alpha = np.einsum('a,aykl->aykl', alpha, Theta)
	
	Theta_matrix = merge_axes(Theta_alpha, 0,2)
	Theta_matrix = merge_axes(Theta_matrix, 1,2)
	
	U, S, V = np.linalg.svd(Theta_matrix)
	
	new_beta2 = split_axes(V, 1, bond_dim, site_dim)[:bond_dim] 
	
	lmbda = S[:bond_dim]
	
	new_beta1 = np.einsum('ayij,kyj->aki', Theta, np.conj(new_beta2))
	
	return new_beta1, lmbda, new_beta2


def apply_single_gate_beta(U, beta1):



	return np.einsum('ayi,ij->ayj', beta1, U)
	
	
	
###################################################
#############  BRANCH DECOMPOSITION ###############
###################################################


#Given a set of square matrices M, outputs a matrix P such that 
#P M P* is in finest block-diagonal decomposition.
#This algorithm from the paper by Maehara and Murota
def block_diagonal_square_matrices(M, epsilon = 1e-5):
	
	#Our M is a 3-tensor representing multiple square matrices we wish 
	#to diagonalize, M(i,j,a), where (i,j) are oredinary indices and 
	#a indexes different matrices.
	
	#Construct the matrices T(i,j)(k,l)
	
	S = np.zeros((M.shape[0], M.shape[0], M.shape[0], M.shape[0]), dtype = np.complex_)
	
	I = np.identity(M.shape[0])
	
	#This is somewhat stupid and numerically inaccurate,
	#probably should update at some point....
	for a in range(M.shape[2]):
		T = np.zeros((M.shape[0], M.shape[0], M.shape[0], M.shape[0]), dtype = np.complex_)
		
		T += np.einsum('qj,ip->ijpq', I, M[:,:,a])
		T -= np.einsum('pi,qj->ijpq', I, M[:,:,a])
		
		T_norm = np.einsum('ijpq, klpq->ijkl', T, np.conj(T))
		
		T_prime = np.zeros((M.shape[0], M.shape[0], M.shape[0], M.shape[0]), dtype = np.complex_)
		
		T_prime += np.einsum('qj,ip->ijpq', I, np.conj(M[:,:,a].T))
		T_prime -= np.einsum('pi,qj->ijpq', I, np.conj(M[:,:,a].T)) 
		
		T_prime_norm = np.einsum('ijpq, klpq->ijkl', T, np.conj(T))
		
		
		S += T_norm
		S += T_prime_norm
	
	S = merge_axes(S, 0, 1)
	S = merge_axes(S, 1, 2)
	
	eigenval, eigenvec = np.linalg.eig(S)
	
	#Find a random Hermitian matrix in the commutant.
	
	small_vecs = [ eigenvec[:,i] for i in range(len(eigenval)) if np.abs(eigenval[i]) < epsilon ]
	
	n_small = len(small_vecs)
	
	
	random_vec = np.random.standard_normal(n_small)
	random_vec = random_vec / np.sqrt(np.sum(random_vec**2))
	
	random_X = np.zeros(M.shape[0]**2, dtype = np.complex_)
	
	for i in range(n_small):
		random_X += random_vec[i] * small_vecs[i]
	
	
	random_X = split_axes(random_X, 0, M.shape[0], M.shape[0])
	
	random_H = (random_X + np.conj(random_X.T) )/2
	
	#This matrix should approximately commute with M[:,:,a]
	
	#for a in range(M.shape[2]):
	#	commute = np.einsum("ij,jk->ik", random_H, M[:,:,a]) - np.einsum("ij,jk->ik", M[:,:,a], random_H)
	#	commute_size = np.sqrt( np.sum(commute**2) )
	#	if commute_size > epsilon:
	#		pdb.set_trace()
	
	
	#Diaogalize this matrix
	W, V = np.linalg.eigh(random_H)
	
	return V
	
#Find  orthonormal bases such that M is in finest block-diagonal form
def block_diagonal_form(M, epsilon = 1e-5):
	
	#M = np.einsum('a,y,ayi->ayi', alpha, eta, M)
	left_matrices = np.einsum('aki,bkj->abij', M, np.conj(M))
	left_matrices = merge_axes(left_matrices, 2, 3)
	
	right_matrices = np.einsum('kai,kbj->abij', np.conj(M), M)
	right_matrices = merge_axes(right_matrices, 2, 3)
	
	#Maybe I should replace this with epsilon**2....
	P = block_diagonal_square_matrices(left_matrices, epsilon)
	Q = block_diagonal_square_matrices(right_matrices, epsilon)
	
	return P, Q
	
	
	
def num_blocks(gamma, epsilon = 1e-5):
	
	P,Q = block_diagonal_form(gamma, epsilon)
	gamma_temp = np.einsum('ab, bci, cd->adi', np.conj(P.T), gamma, Q)
	
	temp_alpha = np.einsum('ayi,ayi->a', gamma_temp, np.conj(gamma_temp))
	temp_eta = np.einsum('ayi,ayi->y', gamma_temp, np.conj(gamma_temp))
	
	blocks = find_blocks(temp_alpha, gamma_temp, temp_eta, epsilon)
	
	return len(blocks)
	
#Returns a list [num_blocks] tallying the number of blocks
#in the finest block decomposition of gamma[i]
def num_blocks_list(gamma_list, lmbda_list, epsilon = 1e-5):
	
	num_blocks = [0]
	
	for i in range(1, len(gamma_list) - 1):
		
		gamma_temp = np.einsum('a,ayi,y->ayi', lmbda_list[i - 1], gamma_list[i], lmbda_list[i])
		bond_dim = gamma_temp.shape[0]
	
		
		P,Q = block_diagonal_form(gamma_temp)
		
		gamma_temp = np.einsum('ab,bci,cd->adi', np.conj(P.T), gamma_temp, Q)
		
		temp_alpha = np.einsum('ayi,ayi->a', gamma_temp, np.conj(gamma_temp))
		temp_eta = np.einsum('ayi,ayi->y', gamma_temp, np.conj(gamma_temp))

		
		blocks = find_blocks(temp_alpha, gamma_temp, temp_eta, epsilon)
		
		num_blocks.append(len(blocks))
		
	num_blocks.append(0)
		
	return num_blocks
		

#Returns the 'block decomposition' of gamma. Projects onto the particular blocks
#at site i, returns lmbda_lists and gamma_lists for each block
def block_decomposition(gamma_list_in, lmbda_list_in, i, epsilon = 1e-5):
	
	gamma_list = copy.deepcopy(gamma_list_in)
	lmbda_list = copy.deepcopy(lmbda_list_in)
	
	gamma_list[i] = np.einsum('a,ayi,y->ayi', lmbda_list[i - 1], gamma_list[i], lmbda_list[i])
	
	bond_dim = gamma_list[i].shape[0]
	
	lmbda_list[i - 1] = np.array([1 for k in range(bond_dim)])
	lmbda_list[i] = np.array([1 for k in range(bond_dim)])
	
	P,Q = block_diagonal_form(gamma_list[i])
	
	gamma_list[i] = np.einsum('ab,bci,cd->adi', np.conj(P.T), gamma_list[i], Q)
	gamma_list[i - 1] = np.einsum('abi,bc->aci', gamma_list[i - 1], P)
	gamma_list[i + 1] = np.einsum('ab,bci->aci', np.conj(Q.T), gamma_list[i + 1])
	
	#Just going to find temporary lmbdas for purposes of block-finding...
	
	temp_alpha = np.einsum('ayi,ayi->a', gamma_list[i], np.conj(gamma_list[i]))
	temp_eta = np.einsum('ayi,ayi->y', gamma_list[i], np.conj(gamma_list[i]))
	
	blocks = find_blocks(temp_alpha, gamma_list[i], temp_eta, epsilon)
	
	decomp_list = []
	for block in blocks:
		gamma_block, lmbda_block = project_onto_block(gamma_list, lmbda_list, block, i)
		decomp_list.append((gamma_block, lmbda_block))
		
	return decomp_list


	
#Given M, returns block decomposition
#in form [blocks] Each entry in blocks is a tuple of form 
#( [3,4,1],[7,5,2] ) for instance, entries in each tuple are coordinates of
#each block on x, y axes respectively. 
def find_blocks(alpha, M, eta, epsilon = 1e-5):
	
	M_norms = np.abs(np.einsum('abi,abi->ab', M, np.conj(M)))
	
	L = M_norms.shape[0]
	x_indices = list(range(L))
	y_indices = list(range(L))
	
	x_indices = [x for x in x_indices if np.abs(alpha[x])**2 > epsilon]
	y_indices = [y for y in y_indices if np.abs(eta[y])**2 > epsilon]
	
	blocks = []
	
	while x_indices != []:
		
		x_unexplored = [x_indices[0]]
		x_indices.pop(0)
		y_unexplored = []
		
		x_explored = []
		y_explored = []
		
		while x_unexplored != [] or y_unexplored != []:
			
			if x_unexplored != []:
			
				x0 = x_unexplored.pop(0)
				x_explored.append(x0)
				
				new_y_indices = [y for y in y_indices if M_norms[x0,y] > epsilon]
				y_unexplored = y_unexplored + new_y_indices
				y_indices = [y for y in y_indices if not(y in new_y_indices)]

			if y_unexplored != []:
			
				y0 = y_unexplored.pop(0)
				y_explored.append(y0)
				
				new_x_indices = [x for x in x_indices if M_norms[x,y0] > epsilon]
				x_unexplored = x_unexplored + new_x_indices
				x_indices = [x for x in x_indices if not(x in new_x_indices)]
				
				
		blocks.append( (x_explored, y_explored) )
		
	return blocks
	
	
	
#Project onto a particular block at site i. block is a tuple of 
#x and y coordinates of the block at site gamma[i]
def project_onto_block(gamma_list_in, lmbda_list_in, block, i):
	
	gamma_list = copy.deepcopy(gamma_list_in)
	lmbda_list = copy.deepcopy(lmbda_list_in)
	
	L = len(gamma_list)
	
	for k in range(len(lmbda_list[i - 1])):
		if not(k in block[0]):
			lmbda_list[i - 1][k] = 0
			
	#lmbda_list[i - 1] = lmbda_list[i - 1] / np.sqrt(np.sum(lmbda_list[i - 1]**2))
			
	for k in range(len(lmbda_list[i])):
		if not(k in block[1]):
			lmbda_list[i][k] = 0
			
	#lmbda_list[i] = lmbda_list[i] / np.sqrt(np.sum(lmbda_list[i]**2))
	
	new_gamma_norm = np.einsum('a,y,ayi,ayi->',lmbda_list[i-1]**2, lmbda_list[i]**2, gamma_list[i], np.conj(gamma_list[i]))
	
	if new_gamma_norm < 1e-12:
		gamma_list[i] = np.zeros(gamma_list[i].shape)
	else:	
		gamma_list[i] = gamma_list[i] / np.sqrt(new_gamma_norm)
	
	#Update left Schmidt decompositions
	
	for k in range(i - 1, -1, -1): 
		gamma_list[k], lmbda_list[k], gamma_list[k + 1] = \
			update_schmidt_left(gamma_list[k], lmbda_list[k], gamma_list[k + 1], lmbda_list[k + 1])
		
	#Right schmidt decompositions
	for k in range(i, L - 1):
		gamma_list[k], lmbda_list[k], gamma_list[k + 1] = \
			update_schmidt_right( lmbda_list[k - 1], gamma_list[k], lmbda_list[k], gamma_list[k + 1])
	
	
	return gamma_list, lmbda_list

	
	