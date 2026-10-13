"""NPU-native drop-in replacements for the torch_scatter ops used in DiffSBDD.

Why: on this Ascend box (torch_npu + torch 2.9), torch_scatter has no NPU
kernels -- calls fall back through the CPU path (the C++ code even asserts
``src must be CPU tensor`` for scatter_max), which copies device<->host and
forces a device sync on every call. That costs ~0.4-1.2 ms per call and
serialises the NPU pipeline once per diffusion step.

The hot path here is deliberately identical in structure to torch_scatter's own
CPU implementation (``drop-in``), except it stays fully on-device:

  * ``dim_size`` derived from ``index.max()`` -- the *single* unavoidable host
    sync per call, exactly what torch_scatter does too;
  * sum via ``torch.Tensor.index_add_``, mean via ``index_add_`` of the values
    and of a ones counter (empty buckets -> 0, matching torch_scatter);
  * indices must be non-negative. DiffSBDD masks are built by
    ``num_nodes_to_batch_mask`` (a ``repeat_interleave`` of 0..n), so they never
    contain -1 -- and torch_scatter's own ``scatter_add_`` would throw on a
    negative index anyway, so the precondition is inherited from the baseline.
    (``scatter_add_masked`` provides the defensive negative-index variant.)
"""
import torch


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    """``torch_scatter.scatter_add`` equivalent (see module docstring)."""
    if out is None:
        shape = list(src.shape)
        if dim_size is None:
            dim_size = int(index.max()) + 1  # one host sync, same as torch_scatter
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.index_add_(dim, index, src)


def scatter_sum(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter_add(src, index, dim, out, dim_size, fill_value)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    """``torch_scatter.scatter_mean`` equivalent (empty buckets -> 0)."""
    if out is None:
        shape = list(src.shape)
        if dim_size is None:
            dim_size = int(index.max()) + 1
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    else:
        dim_size = out.shape[dim]

    summed = out.index_add_(dim, index, src)
    ones = torch.ones(index.numel(), dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, ones)
    denom_shape = [1] * src.dim()
    denom_shape[dim] = -1
    return summed / count.view(denom_shape).clamp(min=1)


def _bcast_mask(valid, dim, src):
    shape = [1] * src.dim()
    shape[dim] = valid.numel()
    return valid.to(src.dtype).view(shape)


def scatter_add_masked(src, index, dim=-1, out=None, dim_size=None):
    """scatter_add that ignores negative indices (defensive variant)."""
    valid = (index >= 0)
    idx = index.clamp(min=0)
    if out is None:
        shape = list(src.shape)
        if dim_size is None:
            dim_size = int(idx[valid].max()).item() + 1 if valid.any() else 0
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    if out.shape[dim] == 0:
        return out
    return out.index_add_(dim, idx, src * _bcast_mask(valid, dim, src))


def scatter_mul(src, index, dim=-1, out=None, dim_size=None, fill_value=1):
    raise NotImplementedError("scatter_mul is not used by DiffSBDD -- add natively if needed")


def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    raise NotImplementedError("scatter_max is not used by DiffSBDD -- add natively if needed")


def scatter_min(src, index, dim=-1, out=None, dim_size=None, fill_value=None):
    raise NotImplementedError("scatter_min is not used by DiffSBDD -- add natively if needed")