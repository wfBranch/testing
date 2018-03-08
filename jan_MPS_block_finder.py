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

    block_shapes = [[1,1,1], [2,2,3], [2,2,3]]
    block_MPS = generate_block_tensor(block_shapes,
                                      block_generator=random_unitary)
    print(block_MPS)

    # Mix up the blocks with a random unitary
    dphys = block_MPS.shape[0]
    dL = block_MPS.shape[1]
    dR = block_MPS.shape[2]
    Vphys = random_unitary((dphys, dphys))
    VL = random_unitary((dL, dL))
    VR = random_unitary((dR, dR))
    block_MPS = np.einsum('aij, ab, il, jk -> blk', block_MPS, Vphys, VL, VR)
    print(block_MPS)

    block_MPS, UR, UL = jan_MPS_block_finder(block_MPS)
    print(block_MPS)

