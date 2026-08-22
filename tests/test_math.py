"""Correctness tests for the two pieces of maths that are easy to get wrong:
the per-head decomposition of the attention output projection, and DARE's
drop-and-rescale being unbiased.

Run with:  pytest -q
"""
import numpy as np
import pytest


def test_head_decomposition_matches_full_projection():
    """sum_h W_O[:, h-block] @ z_h == W_O @ z, i.e. each head's contribution to
    the residual stream is exactly its own column block of W_O."""
    rng = np.random.default_rng(0)
    d_model, n_heads, d_head = 128, 8, 16
    W = rng.normal(size=(d_model, n_heads * d_head))
    z = rng.normal(size=(n_heads * d_head,))

    full = W @ z
    per_head = sum(W[:, h * d_head:(h + 1) * d_head] @ z[h * d_head:(h + 1) * d_head]
                   for h in range(n_heads))
    assert np.allclose(full, per_head, atol=1e-10)


def test_patching_pre_projection_equals_patching_post_projection():
    """Replacing z_h with zbar_h before W_O is equivalent to replacing the head's
    projected contribution — which is why the patcher can hook o_proj's input."""
    rng = np.random.default_rng(1)
    d_model, n_heads, d_head, target = 128, 8, 16, 3
    W = rng.normal(size=(d_model, n_heads * d_head))
    z = rng.normal(size=(n_heads * d_head,))
    zbar = rng.normal(size=(d_head,))

    patched_input = z.copy()
    patched_input[target * d_head:(target + 1) * d_head] = zbar
    lhs = W @ patched_input

    blk = W[:, target * d_head:(target + 1) * d_head]
    rhs = (W @ z) - blk @ z[target * d_head:(target + 1) * d_head] + blk @ zbar
    assert np.allclose(lhs, rhs, atol=1e-10)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7])
def test_dare_rescaling_is_unbiased(p):
    """E[mask * delta / (1-p)] == delta, which is the whole point of the rescale."""
    rng = np.random.default_rng(2)
    delta = rng.normal(size=(400_000,))
    mask = (rng.random(delta.shape) >= p).astype(float)
    dare = mask * delta / (1.0 - p)
    assert abs(dare.mean() - delta.mean()) < 0.01
    sparsity = float((dare == 0).mean())
    assert abs(sparsity - p) < 0.01


def test_safety_vector_is_negated_harmful_direction():
    """theta_sft + (theta_base - theta_harmful) equals the task-arithmetic form
    theta_base + (theta_sft - theta_base) - (theta_harmful - theta_base)."""
    rng = np.random.default_rng(3)
    base, sft, harmful = (rng.normal(size=(64,)) for _ in range(3))
    direct = sft + (base - harmful)
    task_arithmetic = base + (sft - base) - (harmful - base)
    assert np.allclose(direct, task_arithmetic)


def test_fewshot_pools_are_disjoint():
    torch = pytest.importorskip("torch")  # prompts module pulls in the stack
    from safealign.fv.prompts import build_fewshot_pairs
    pairs = build_fewshot_pairs(n_prompts=5, n_icl=4, seed=7)
    targets = {p.idx for p in pairs}
    icl = {i for p in pairs for i in p.icl_indices}
    assert not (targets & icl)
