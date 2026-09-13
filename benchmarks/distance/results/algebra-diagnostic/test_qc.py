import numpy as np
import pytest
from qc_module import field_rank,poly_div,poly_mul
from run import gf2

@pytest.mark.parametrize('modulus',[7,11])
def test_field_rank_matches_binary_multiplication_representation(modulus):
    d=modulus.bit_length()-1;rng=np.random.default_rng(93)
    for _ in range(12):
        matrix=rng.integers(0,1<<d,(3,4)).tolist()
        expanded=np.zeros((3*d,4*d),dtype=np.int8)
        for i,row in enumerate(matrix):
            for j,value in enumerate(row):
                for k in range(d):
                    product=poly_div(poly_mul(value,1<<k),modulus)[1]
                    for bit in range(d):expanded[i*d+bit,j*d+k]=(product>>bit)&1
        assert gf2.rank(expanded)==d*field_rank(matrix,modulus)
