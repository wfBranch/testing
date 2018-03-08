import numpy as np

def complex_randn(shape):
    """ `M = complex_randn(shape)`
    Return a complex matrix of dimensions `shape`, with random
    elements sampled from a Gaussian distribution centered at 0 and with
    variance 1.
    """
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)


def random_unitary(shape):
    """ `U = random_unitary(shape)`
    If `shape` is a shape of a matrix (len = 2), then `U` is a
    unitary/isometry of this shape, generated from the SVD of a
    Gaussian-random complex matrix. If `shape` has len = 3, U is of
    `shape`, and `U[i,:,:]` is a random unitary/isometric matrix for
    each `i`. For `len(shape)` != 2 or 3, an error is rised.
    """
    if len(shape) == 2:
        # We need just a matrix.
        U = np.linalg.svd(complex_randn(shape))[0]
    elif len(shape) == 3:
        # We need a tensor.
        U = np.empty(shape, dtype=np.complex_)
        matrix_shape = (shape[-2], shape[-1])
        for i in range(shape[0]):
            U[i,:,:] = random_unitary(matrix_shape)
    else:
        msg = "Can't generate random unitaries with shape {}.".format(shape)
        raise ValueError(msg)
    return U


def generate_block_tensor(leg_dims, block_generator=complex_randn):
    """
    `T = generate_block_tensor(leg_dims, block_generator=complex_randn)`
    Generate a tensor `T` with block structure. `leg_dims` should be a
    list with one element for each leg of `T`. For each leg `i`,
    `leg_dims[i]` should be either
    a) an integer, in which case there is no block structure on leg `i`
    leg, and all blocks have dimension `leg_dims[i]` along leg `i`, and
    thus span the whole tensor,
    or
    b) a list of integers, that lists the dimensions of the various
    blocks along leg `i`.

    As an example, `leg_dims = [[2,4], 5, [2,6]]` would generate a
    tensor `T` with the shape (6,5,8), that would consist of two blocks
    of shapes (2,5,2) and (4,5,6).
    
    Note that all elements of `leg_dims` should either have the same
    length, or be scalars, othewise behavior is undefined.

    The keyword argument `block_generator` takes a function that takes
    in a shape and return a tensor of this shape. This function
    generates the blocks.  By default it generates random complex blocks
    with Gaussian distributions.
    """
    R = len(leg_dims)  # Rank of the tensor.
    B = max(map(len, leg_dims))  # Number of blocks.
    total_dims = tuple(map(sum, leg_dims))  # Total bond dimensions.
    T = np.zeros(total_dims, dtype=np.complex_)
    sum_dims = [0]*R  # Keeps track of how far we've gone along each leg.
    for b in range(B):
        shp = [None]*R  # A placeholder for the shape of this block.
        slices = [None]*R  # A placeholder for the slice into this block.
        for r in range(R):
            dimsr = leg_dims[r]
            if hasattr(dimsr, "__getitem__"):  # This leg has block structure.
                db = dimsr[b]
                shp[r] = db
                slices[r] = slice(sum_dims[r], sum_dims[r]+db)
                sum_dims[r] += db
            else:  # This leg only has a single dimension across all blocks.
                db = dimsr
                shp[r] = db
                slices[r] = slice(None)
        block = block_generator(shp)
        T.__setitem__(slices, block)  # Same as T[*slices] = block
    return T
        


