import numpy as np


"""
An implementation of the block finding algorithm that's based on
contracting with a random operator on one side, and tracing over the middle.
"""


def complex_randn_hermitian(shape):
    """ `H = complex_randn_hermitian(shape)`
    Return a random complex Hermitian matrix.
    """
    H = np.random.randn(*shape) + 1j*np.random.randn(*shape)
    H = (H + H.conjugate().transpose())/2
    return H


def jan_MPS_block_finder(Gamma):
    """ `Gammaprime, UR, UL = jan_MPS_block_finder(Gamma)`
    Takes in an MPS tensor `Gamma`, with the first index being the
    physical one, that presumably has block structure between the three
    indices, but possibly mixed up by gauge transformations.
    `jan_MPS_block_finder` then finds the gauge transformations `UR` and
    `UL` for the virtual indices that make the blocks manifest, and
    returns them together with the gauge-transformed tensor
    `Gammaprime`. If no block structure is found, `UR` and `UL` are
    arbitrary gauge transformations.
    """
    DL = Gamma.shape[1]
    DR = Gamma.shape[2]

    L_op = complex_randn_hermitian((DL, DL))
    R = np.einsum('ik, aim, akn -> mn', L_op, Gamma, Gamma.conjugate())
    SR, UR = np.linalg.eigh(R)
    UR = UR.conjugate()
    # TODO Add a warning if SR has degenerate values.

    R_op = complex_randn_hermitian((DR, DR))
    L = np.einsum('aim, akn, mn -> ik', Gamma, Gamma.conjugate(), R_op)
    SL, UL = np.linalg.eigh(L)
    UL = UL.conjugate()
    # TODO Add a warning if SL has degenerate values.

    Gamma = np.einsum("aij, ik, jl -> akl", Gamma, UL, UR)
    return Gamma, UR, UL


if __name__ == "__main__":
    # If this module is called as a script, run a test case.
    from generate_block_tensor import generate_block_tensor, random_unitary
    np.set_printoptions(suppress=True, linewidth=120)

    noise_eps = 1e-6
    block_shapes = [[1,1,1], [2,2,3], [2,2,3]]
    block_MPS = generate_block_tensor(block_shapes,
                                      block_generator=random_unitary)
    block_MPS_noise = block_MPS + noise_eps*np.ones_like(block_MPS)

    GHZ_dim = 3
    GHZ = np.zeros((GHZ_dim,)*3, dtype=np.complex_)
    for i in range(GHZ_dim):
        GHZ[i,i,i] = 1.
    GHZ_noise = GHZ + noise_eps*np.ones_like(GHZ)

    Gammas = [GHZ, GHZ_noise, block_MPS, block_MPS_noise]
    for Gamma in Gammas:
        print("="*70)
        print(Gamma)

        # Mix up the blocks with random unitaries.
        dphys = Gamma.shape[0]
        dL = Gamma.shape[1]
        dR = Gamma.shape[2]
        Vphys = random_unitary((dphys, dphys))
        VL = random_unitary((dL, dL))
        #VR = random_unitary((dR, dR))
        VR = VL.conjugate()
        Gamma = np.einsum('aij, ab, il, jk -> blk', Gamma, Vphys, VL, VR)
        print("- "*30)
        print(Gamma)

        Gamma, UR, UL = jan_MPS_block_finder(Gamma)
        print("- "*30)
        print(Gamma)

