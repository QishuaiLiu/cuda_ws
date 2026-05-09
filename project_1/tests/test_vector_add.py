import numpy as np
import pytest

import _project_1_py as p1


@pytest.mark.parametrize("n", [1, 5, 256, 257, 1023, 4096, 100_003])
def test_matches_numpy(n):
    rng = np.random.default_rng(seed=n)
    a = rng.standard_normal(n, dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)
    got = p1.vector_add(a, b)

    assert got.dtype == np.float32
    assert got.shape == (n,)
    np.testing.assert_allclose(got, a + b, rtol=0, atol=1e-6)


def test_does_not_alias_inputs():
    a = np.ones(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32) * 2
    out = p1.vector_add(a, b)
    out[0] = 999.0
    assert a[0] == 1.0 and b[0] == 2.0


def test_shape_mismatch_raises():
    with pytest.raises(RuntimeError, match="same length"):
        p1.vector_add(np.zeros(3, np.float32), np.zeros(4, np.float32))


def test_wrong_dtype_rejected():
    a = np.zeros(4, dtype=np.float64)
    b = np.zeros(4, dtype=np.float64)
    with pytest.raises(TypeError):
        p1.vector_add(a, b)


def test_non_contiguous_rejected():
    base = np.arange(16, dtype=np.float32)
    a = base[::2]   # strided, not c_contig
    b = base[::2]
    with pytest.raises(TypeError):
        p1.vector_add(a, b)
