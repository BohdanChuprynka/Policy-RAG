import numpy as np

from policy_app.llm.embedding import l2_normalize

def test_l2_normalize_unit_norm_rows():
    mat = np.array([[3.0, 4.0, 0.0],
                    [1.0, 2.0, 2.0]], dtype=np.float32)

    out = l2_normalize(mat)

    assert out.shape == (2, 3)

    # Norms check 
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)

def test_l2_normalize_zero_row_no_nans():
    mat = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]], dtype=np.float32)

    out = l2_normalize(mat)

    assert not np.any(out[0])
    assert not np.any(np.isnan(out)) and not np.any(np.isinf(out))

def test_l2_normalize_preserves_dtype():
    mat = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    out = l2_normalize(mat)

    assert out.dtype == np.float32
