import random

import numpy as np
from ecmm.foo import bar


def test_rejectionsampling():
    random.seed(2023)
    np.array([[-1, 0.5, 0.5], [0.5, -1, 0.5], [0.5, 0.5, -1]])
    assert 1 == bar()
    # result = modifiedforward(bgnst, endst, Q, 10)
    # assert isinstance(result, Path)
    # assert len(result.array_a) == len(result.array_b)
    # assert result.array_a[0] == bgnst
    # assert result.array_a[-1] == endst
