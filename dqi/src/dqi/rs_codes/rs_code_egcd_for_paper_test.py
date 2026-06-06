import numpy as np
import pytest
import galois
from galois import GF

from dqi.rs_codes.rs_code_egcd_for_paper import (
    rs_syndrome_decoder_eea_basic,
    rs_syndrome_decoder_eea_divstep_expanded,
    rs_syndrome_decoder_eea_divstep,
    rs_syndrome_decoder_eea_divstep_opt,
    rs_syndrome_decoder_eea_zalka,
)


def generate_msg_and_codewords(
    rs, block_length: int, dimension: int, n_iterations: int, n_errors: int
):
    gf_type = GF(block_length + 1)
    msgs = gf_type.Random((n_iterations, dimension), low=1, seed=rs)
    gamma = gf_type.primitive_element
    B = gf_type([[gamma ** (i * j) for j in range(dimension)] for i in range(block_length)])
    codewords = (B @ msgs.transpose()).transpose()
    errors = gf_type.Random((n_iterations, block_length), low=1, seed=rs)
    error_inds = rs.choice(np.arange(block_length), size=(n_iterations, n_errors))
    add_errors = np.zeros((n_iterations, block_length), dtype=int)
    add_errors[np.indices((n_iterations, n_errors))[0], error_inds] += 1
    assert np.all(np.sum(add_errors, axis=1) < (block_length - dimension + 1) // 2)
    P = gf_type(
        [
            [gamma ** (i * j) for j in range(block_length)]
            for i in range(1, block_length - dimension + 1)
        ]
    )
    syndromes = (P @ (add_errors * errors).transpose()).transpose()
    return msgs, codewords, codewords + add_errors * errors, syndromes


@pytest.mark.parametrize(
    'n, k, n_errors',
    [(31, 15, 1), (31, 15, 3), (31, 15, 5), (31, 15, 7), (31, 7, 10)]
    + [(63, 29, 16), (127, 61, 32), (255, 125, 64)],
)
@pytest.mark.parametrize(
    'syndrome_decoder',
    [
        rs_syndrome_decoder_eea_basic,
        rs_syndrome_decoder_eea_divstep_expanded,
        rs_syndrome_decoder_eea_divstep,
        rs_syndrome_decoder_eea_divstep_opt,
        rs_syndrome_decoder_eea_zalka,
    ],
)
@pytest.mark.parametrize('seed', range(1, 11))
def test_rs_decoding_via_eea_syndrome_decoder(
    n: int, k: int, n_errors: int, syndrome_decoder, seed: int
):
    rs = np.random.default_rng(seed)
    msgs, codewords, received, syndromes = generate_msg_and_codewords(rs, n, k, 50, n_errors)
    for i, (msg, codeword, rec, syndrome) in enumerate(zip(msgs, codewords, received, syndromes)):
        error_pattern = syndrome_decoder(syndrome)
        assert np.all(rec == codeword + error_pattern)
