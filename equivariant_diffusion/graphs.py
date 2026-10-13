"""L5: static edge tables + in-graph dynamic edge masks.

L4 captured the dynamics step but still built the real edge list eagerly
every step (get_edges: cdist + nonzero + variable-shape copies). Profile at
n=20: that eager extraction cost 6.83s/round (13.6ms/step) vs 0.28s
(0.57ms/step) for the replay itself, i.e. ~75% of the run. The nonzero shape
sync means it can never go inside a graph -- so L5 removes it entirely.

Design (per the user's direction):
  * edge construction moves OUT of the graph completely: once per round we
    build three *static* pair tables, ordered by the two link categories that
    vary and the one that doesn't:
        - lig-lig   (LL):   every same-molecule ordered pair incl. diag
        - cross     (X):    every same-copy ligand x pocket pair
        - poc-poc   (PP):   same-copy pocket pairs, pre-pruned to the real
                            edges by the *static* pocket geometry (mask 1)
  * INSIDE the graph only fixed-shape *dynamic mask* computation happens:
    LL/X masks are (dist^2 <= cutoff^2) evaluated on the static tables from
    the current coordinate buffer each replay; PP contributes constant 1.
    All three are concatenated into the existing ``edge_mask`` column vector
    that egnn already multiplies into node/coord messages, so non-edges
    contribute exactly 0 (= the eager semantics, including self-loops, which
    eager nonzero also produced since dist=0 <= cutoff).

Per step outside the graph: only two buffer copies + replay.
Node counts are round-constant in this sampling scheme, so one capture per
round; re-capture here would re-run _build_tables + _capture.
"""
import os
import time

import torch
from scatter_ops import scatter_mean


def compute_edge_mask(x_cat, be, segs):
    """Fixed-shape dynamic masks over the static pair tables (L5).

    Returns the (E, 1) float column vector consumed by egnn's edge_mask:
    1.0 for real edges (same-molecule pair within cutoff), 0.0 else. PP is a
    pre-pruned static table so its slice is constant 1. This runs inside the
    captured graph on the current coordinate buffer, but is a plain function
    so the same expressions can be re-evaluated eagerly for verification.
    """
    if os.environ.get('V0_V1MASK') == '1':
        return compute_edge_mask_v1(x_cat, be, segs)
    parts = []
    for name, lo, hi, cutoff in segs:
        if cutoff is not None and name != 'pp':
            r = be[0, lo:hi]
            c = be[1, lo:hi]
            diff = x_cat[r] - x_cat[c]
            d2 = (diff * diff).sum(dim=1, keepdim=True)
            parts.append((d2 <= cutoff * cutoff).float())
        else:
            parts.append(torch.ones((hi - lo, 1), device=x_cat.device))
    return torch.cat(parts, dim=0)


def compute_edge_mask_v1(x_cat, be, segs):
    """v1-parity mask: ONE gather pair over the whole dynamic slice (ll+x),
    per-edge threshold array, static PP as constant ones at the end. This
    mirrors v1's aclgraph_step.forward path exactly (fewer kernels on E)."""
    dyn = segs[1][2]  # hi of the x segment = ll + x
    r = be[0, :dyn]
    c = be[1, :dyn]
    diff = x_cat[r] - x_cat[c]
    d2 = (diff * diff).sum(dim=-1)

    def thr_sq(cut):
        return float('inf') if cut is None else float(cut) * float(cut)

    t_ll = torch.full((segs[1][1],), thr_sq(segs[0][3]), device=x_cat.device,
                      dtype=x_cat.dtype)
    t_x = torch.full((segs[1][2] - segs[1][1],), thr_sq(segs[1][3]),
                     device=x_cat.device, dtype=x_cat.dtype)
    dyn_mask = (d2 <= torch.cat([t_ll, t_x])).to(x_cat.dtype)
    static = torch.ones(segs[2][2] - dyn, device=x_cat.device, dtype=x_cat.dtype)
    return torch.cat([dyn_mask, static], dim=0).unsqueeze(-1)


def build_edge_tables(dynamics, mask_atoms, mask_residues, x_lig, x_poc):
    """L5 static pair tables for the three edge categories (once per round).

    Returns (tables (2, E) long on device, segs [(name, lo, hi, cutoff)],
    types (E,) long with aclnn edge-type ids 1/0/2).
    """
    dev = x_lig.device
    N_lig = mask_atoms.shape[0]
    N_poc = mask_residues.shape[0]
    off_poc = N_lig  # node-index offset of pocket in the combined space

    def ordered_pairs(adj, row_off=0, col_off=0):
        r, c = adj.nonzero(as_tuple=True)  # variable shape is fine here
        return torch.stack([r + row_off, c + col_off], dim=0)

    parts = []
    segs = []

    # LL: all same-molecule ordered pairs, incl. diagonal (eager cdist also
    # kept self-loops: dist 0 <= cutoff). Cutoff applied as in-graph mask.
    adj_ll = mask_atoms[:, None] == mask_atoms[None, :]
    e_ll = ordered_pairs(adj_ll)
    P_ll = e_ll.size(1)
    parts.append(e_ll)
    segs.append(('ll', 0, P_ll, dynamics.edge_cutoff_l))

    # X: same-copy ligand x pocket pairs. Eager emits BOTH directions
    # (adj_cross upper-right and adj_cross.T lower-left in the full adj),
    # so enumerate the table transposed too.
    adj_x = mask_atoms[:, None] == mask_residues[None, :]
    e_x = torch.cat([
        ordered_pairs(adj_x, row_off=0, col_off=off_poc),
        ordered_pairs(adj_x.t(), row_off=off_poc, col_off=0),
    ], dim=1)
    P_x = e_x.size(1)
    parts.append(e_x)
    segs.append(('x', P_ll, P_ll + P_x, dynamics.edge_cutoff_i))

    # PP: same-copy pocket pairs, pruned to the real edges by the *static*
    # pocket geometry (mask constant 1 in-graph). Self-loops kept.
    adj_pp = mask_residues[:, None] == mask_residues[None, :]
    if dynamics.edge_cutoff_p is not None:
        adj_pp = adj_pp & (torch.cdist(x_poc, x_poc) <= dynamics.edge_cutoff_p)
    e_pp = ordered_pairs(adj_pp, row_off=off_poc, col_off=off_poc)
    P_pp = e_pp.size(1)
    parts.append(e_pp)
    segs.append(('pp', P_ll + P_x, P_ll + P_x + P_pp, dynamics.edge_cutoff_p))

    e = torch.cat(parts, dim=1).contiguous()
    Etot = e.size(1)
    # type ids follow the original semantics 0=cross, 1=lig-lig, 2=pocket-pocket
    bet = torch.zeros(Etot, dtype=torch.long, device=dev)
    bet[segs[0][1]:segs[0][2]] = 1
    bet[segs[2][1]:segs[2][2]] = 2
    return e.to(dev), segs, bet


def build_edge_tables_reduced(dynamics, mask_atoms, mask_residues,
                              x_lig, x_poc, margin=0.75):
    """L8 trapezoid: build_edge_tables, but with the ll/x rows pruned by
    dist <= cutoff+margin on the *current* coordinates at re-capture time.

    Coverage contract: the row set of each ll/x slice here is a *subset* of
    the full table's slice (same enumeration order, extra distance bound), so
    the hook `_recapture` can verify real-edge-set equality against the
    outgoing table by comparing mask sums. The +margin guards edges that the
    remaining <simga_s> trajectory can bring into the cutoff; sigma_s at the
    re-capture point is logged for tuning L8_MARGIN. PP stays static (same
    pairs as the full table: pocket geometry is round-constant).
    """
    dev = x_lig.device
    N_lig = mask_atoms.shape[0]
    off_poc = N_lig

    def ordered_pairs(adj, row_off=0, col_off=0):
        r, c = adj.nonzero(as_tuple=True)
        return torch.stack([r + row_off, c + col_off], dim=0)

    parts = []
    segs = []

    cutoff_l = dynamics.edge_cutoff_l
    cutoff_i = dynamics.edge_cutoff_i
    if cutoff_l is not None:
        adj_ll = (mask_atoms[:, None] == mask_atoms[None, :]) & (
            torch.cdist(x_lig, x_lig) <= cutoff_l + margin)
    else:
        adj_ll = mask_atoms[:, None] == mask_atoms[None, :]
    e_ll = ordered_pairs(adj_ll)
    P_ll = e_ll.size(1)
    parts.append(e_ll)
    segs.append(('ll', 0, P_ll, cutoff_l))

    adj_x = mask_atoms[:, None] == mask_residues[None, :]
    if cutoff_i is not None:
        adj_cross = adj_x & (torch.cdist(x_lig, x_poc) <= cutoff_i + margin)
    else:
        adj_cross = adj_x
    e_x = torch.cat([
        ordered_pairs(adj_cross, row_off=0, col_off=off_poc),
        ordered_pairs(adj_cross.t(), row_off=off_poc, col_off=0),
    ], dim=1)
    P_x = e_x.size(1)
    parts.append(e_x)
    segs.append(('x', P_ll, P_ll + P_x, cutoff_i))

    adj_pp = mask_residues[:, None] == mask_residues[None, :]
    if dynamics.edge_cutoff_p is not None:
        adj_pp = adj_pp & (torch.cdist(x_poc, x_poc) <= dynamics.edge_cutoff_p)
    e_pp = ordered_pairs(adj_pp, row_off=off_poc, col_off=off_poc)
    P_pp = e_pp.size(1)
    parts.append(e_pp)
    segs.append(('pp', P_ll + P_x, P_ll + P_x + P_pp, dynamics.edge_cutoff_p))

    e = torch.cat(parts, dim=1).contiguous()
    Etot = e.size(1)
    bet = torch.zeros(Etot, dtype=torch.long, device=dev)
    bet[segs[0][1]:segs[0][2]] = 1
    bet[segs[2][1]:segs[2][2]] = 2
    return e.to(dev), segs, bet


class DynGraph:
    """Capture dynamics.forward once per round over static edge tables and
    replay each denoising step; the per-step edge masks are computed inside
    the graph from the current coordinate buffers (fixed shape)."""

    def __init__(self, dynamics, headroom=0.15, min_grow=32):
        self.dynamics = dynamics
        self.headroom = headroom
        self.min_grow = min_grow
        self._g = None
        self._buf = None
        self._outs = None
        self.n_captures = 0
        self._te = 0.0
        self._tb = 0.0
        self._step_count = 0
        # diagnostics computed at capture (E_max-equivalent counts)
        self.table_sizes = None

    @property
    def active(self):
        return self._g is not None

    def step(self, z, p, t, mask_atoms, mask_residues):
        _pb = time.perf_counter()
        if self._g is None:
            self._capture(z, p, t, mask_atoms, mask_residues)
            self.n_captures += 1
        t0 = time.perf_counter()
        b = self._buf
        b['z'].copy_(z)
        b['p'].copy_(p)
        b['t'].copy_(t)
        self._g.replay()
        t1 = time.perf_counter()
        if os.environ.get('DYN_TIMER'):
            self._te += t1 - t0
            self._tb += t0 - _pb
            self._step_count += 1
        return self._outs

    # ------------------------------------------------------------------
    # table construction (eager, once per round, outside the graph)
    # ------------------------------------------------------------------
    def _build_tables(self, mask_atoms, mask_residues, x_lig, x_poc):
        e, segs, _ = build_edge_tables(
            self.dynamics, mask_atoms, mask_residues, x_lig, x_poc)
        self.table_sizes = {'ll': segs[0][2], 'x': segs[1][2] - segs[1][1],
                            'pp': segs[2][2] - segs[2][1],
                            'total': e.size(1),
                            'n_lig': x_lig.shape[0], 'n_poc': x_poc.shape[0]}
        return e, segs

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------
    def _capture(self, z, p, t, mask_atoms, mask_residues):
        dev = z.device
        d = self.dynamics
        n_dims = d.n_dims

        x_lig, x_poc = z[:, :n_dims], p[:, :n_dims]
        be, segs, bet = build_edge_tables(
            d, mask_atoms, mask_residues, x_lig, x_poc)
        self.table_sizes = {'ll': segs[0][2],
                            'x': segs[1][2] - segs[1][1],
                            'pp': segs[2][2] - segs[2][1],
                            'total': be.size(1),
                            'n_lig': x_lig.shape[0], 'n_poc': x_poc.shape[0]}

        bz = torch.empty_like(z)
        bp = torch.empty_like(p)
        bt = torch.empty_like(t)
        self._buf = {'z': bz, 'p': bp, 't': bt}

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            # dynamic masks, fixed shape, computed from the *current* coords
            x_cat = torch.cat([bz[:, :n_dims], bp[:, :n_dims]], dim=0)
            edge_mask = compute_edge_mask(x_cat, be, segs)  # (Etot, 1)

            out_lig, out_poc = d(
                bz, bp, bt, mask_atoms, mask_residues,
                precomputed_edges=be, precomputed_edge_types=bet,
                precomputed_edge_mask=edge_mask)
        self._g = g
        self._outs = (out_lig, out_poc)
        if not getattr(self, '_quiet', False) and os.environ.get('DYN_TIMER'):
            ts = self.table_sizes
            print(f"[DynGraph] tables ll={ts['ll']} x={ts['x']} pp={ts['pp']} "
                  f"tot={ts['total']} (n_lig={ts['n_lig']} n_poc={ts['n_poc']})")
        if os.environ.get('L5_CHECK'):
            self._verify_membership(z, p, mask_atoms, mask_residues, be, segs)

    def _verify_membership(self, z, p, mask_atoms, mask_residues, be, segs):
        """One-shot correctness anchor (L5_CHECK=1): the in-graph mask scheme
        must select exactly the same (src, dst) pairs as the eager path."""
        d = self.dynamics
        n_dims = d.n_dims
        x_lig, x_poc = z[:, :n_dims], p[:, :n_dims]
        eager_edges = d.get_edges(mask_atoms, mask_residues, x_lig, x_poc)
        x_cat = torch.cat([x_lig, x_poc], dim=0)
        m = compute_edge_mask(x_cat, be, segs)
        masked = (m[:, 0] > 0).nonzero(as_tuple=True)[0]
        dense_edges = be[:, masked].cpu().numpy()
        eager_set = set(map(tuple, eager_edges.t().tolist()))
        dense_set = set(map(tuple, dense_edges.T.tolist()))
        only_eager = eager_set - dense_set
        only_dense = dense_set - eager_set
        ok = not only_eager and not only_dense
        print(f"[L5 check] eager={len(eager_set)} dense={len(dense_set)} "
              f"only_eager={len(only_eager)} only_dense={len(only_dense)} "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            n_lig = x_lig.shape[0]
            ma = mask_atoms.cpu().numpy()
            mr = mask_residues.cpu().numpy()
            import numpy as np
            xcat = x_cat
            for (r, c) in list(only_eager)[:12]:
                if r < n_lig and c < n_lig:
                    cat = f"LL same_mol={ma[r]==ma[c]}"
                elif r < n_lig:
                    cat = f"X   copy_eq={ma[r]==mr[c-n_lig]}"
                else:
                    cat = "PP"
                d2 = float(((xcat[r]-xcat[c])**2).sum())
                in_tbl = set(map(tuple, be.t().tolist()))
                print(f"  only_eager ({r},{c}) {cat} d2={d2:.3f} "
                      f"cutoff2={25.0} in_table={ (r,c) in in_tbl }", flush=True)
            print("  masks ligand:", np.unique(ma), "pocket:", np.unique(mr),
                  flush=True)
            raise AssertionError(
                f"membership mismatch: eager-only {list(only_eager)[:5]}")

class AclGraphStepWrapper:
    """L6: one NPU graph for a *whole* denoising step.

    L5 captured only the EGNN; per step the diffusion math around it (gamma,
    sigma2/alpha_t_given_s, mu, noise injection, remove_mean COM projection)
    still ran eager -- ~40 host-dispatched ops/step and a nonzero-shape sync
    inside them when they recomputed edges... no, edges are gone; the residual
    eager cost was pure host dispatch of the outer physics. Profiling L5 at
    n=20 showed ~4ms/step of that outer eager work vs ~7ms/step of replay.

    This wrapper folds the outer physics INTO the same captured graph by
    buffering every per-step-varying input:
      z_buf / p_buf   : current state, rewritten *in place* by the graph
                        (read-at-top / write-at-end, so the loop variable and
                        the graph's input are the same tensor)
      s_buf (int, n,1): step index; host does one fill_() per step; the graph
                        computes s/t normalised *on device* with the exact
                        eager arithmetic (float(s)/timesteps etc.)
      e_buf           : injected Gaussian noise -- host fills via .normal_()
                        each step. randn is deliberately NOT inside the graph
                        so every replay sees fresh noise (a captured randn
                        would reuse the same draw).

    Sequence inside the graph, mirroring sample_p_zs_given_zt exactly:
      s,t -> gamma_s, gamma_t -> sigma2/sigma/alpha_t_given_s + sigma_s/t
      -> EGNN on static edge tables + in-graph dynamic masks (L5 machinery)
      -> mu = z/a_t_given_s[mask] - (sigma2/a/sigma_t)[mask] * eps
      -> zs = mu + sigma_full[mask] * e_buf -> remove_mean (lig AND pocket)
      -> write back into z_buf/p_buf (in place). Host per step: 2 fills+replay.

    Edge tables are built once per round (node counts are round-constant).
    """
    def __init__(self, model, timesteps):
        self.model = model
        self.dynamics = model.dynamics
        self.timesteps = timesteps
        self._g = None
        self._buf = None
        self.n_captures = 0
        self.table_sizes = None
        self._te = 0.0
        self._tb = 0.0
        self._step_count = 0
        self._mask_atoms = None
        self._mask_residues = None
        self._n_dims = 0
        self._n_samples = 1
        self._recaps = []

    @property
    def active(self):
        return self._g is not None

    def wrap(self, z, p, mask_atoms, mask_residues):
        """Copy the starting state into buffers and capture the step graph.
        Returns (z_buf, p_buf) -- the loop then iterates these in place.

        L8: also keep the wrap-time FULL edge tables aside as self._be0, and
        parse L8_RECAP (descending step list, "450,300,150") for mid-round
        trapezoid re-captures. The full tables are unconditionally reused by
        the final decode (post-loop exact semantics), so re-captures only
        ever shrink the in-round EGNN work, never the decode's coverage.
        """
        dev = z.device
        d = self.dynamics
        n_dims = d.n_dims
        n_samples = self.model._round_n_samples
        model = self.model
        T = self.timesteps
        self._mask_atoms = mask_atoms
        self._mask_residues = mask_residues
        self._n_dims = n_dims
        self._n_samples = n_samples

        z_buf = torch.empty_like(z)
        z_buf.copy_(z)
        p_buf = torch.empty_like(p)
        p_buf.copy_(p)
        s_buf = torch.zeros((n_samples, 1), dtype=torch.float32, device=dev)
        e_buf = torch.zeros_like(z)
        self._buf = {'z': z_buf, 'p': p_buf, 's': s_buf, 'e': e_buf}

        x_lig, x_poc = z[:, :n_dims], p[:, :n_dims]
        be, segs, bet = build_edge_tables(d, mask_atoms, mask_residues,
                                          x_lig, x_poc)
        self._be0, self._segs0, self._bet0 = be, segs, bet
        self.table_sizes = {'ll': segs[0][2],
                            'x': segs[1][2] - segs[1][1],
                            'pp': segs[2][2] - segs[2][1],
                            'total': be.size(1),
                            'n_lig': x_lig.shape[0], 'n_poc': x_poc.shape[0]}
        self._twrap = time.perf_counter()

        # L6: the per-replay edge canary is constructional (mask / static
        # edge tables are round-constant), so validate it eagerly HERE, before
        # capture, and keep the torch.all op OUT of the captured graph --
        # reading a graph-baked tensor at round end costs ~2.5s on torch_npu.
        if d._round_edges_ok is not None:
            d._round_edges_checked = True
            self._set_eager_canary(be)

        _nwarm = int(os.environ.get('L7_WARMUP', '1'))
        _tww = self._warmup_chain(be, segs, bet, z, p, _nwarm, label='wrap')
        _tw0 = time.perf_counter()
        self._g = self._capture_chain(be, segs, bet)
        _tcap = time.perf_counter() - _tw0

        self._be, self._segs, self._bet = be, segs, bet
        self.n_captures += 1
        self._twrap = time.perf_counter() - self._twrap
        self._tww = _tww
        self._sync = 0.0
        self._sync_host = 0.0
        _recap_env = os.environ.get('L8_RECAP', '')
        self._recaps = sorted(
            (int(x) for x in _recap_env.replace(';', ',').split(',')
             if x.strip()), reverse=True) if _recap_env.strip() else []
        if os.environ.get('DYN_TIMER'):
            print(f"[wrap] tables+fill={self._twrap:.3f}s "
                  f"(warmup={self._tww:.3f}s "
                  f"capture-blk={_tcap:.3f}s) recaps={self._recaps}")
        # The capture itself executed the graph once, and its in-place writes
        # at the end have already advanced z_buf/p_buf by one garbage step.
        # Reset to the true round-start state so the loop's first replay()
        # iterates from z0, not from that phantom step.
        z_buf.copy_(z)
        p_buf.copy_(p)
        return z_buf, p_buf

    def _set_eager_canary(self, be):
        """Round-eager edge canary for a given table (torch.all stays OUT of
        the captured graph; read back eagerly at round end)."""
        _mask = torch.cat([self._mask_atoms, self._mask_residues])
        self.dynamics._round_eager_edges_ok = torch.all(
            _mask[be[0]] == _mask[be[1]])

    def _warmup_chain(self, be, segs, bet, z_ref, p_ref, nwarm, label='warm'):
        """L7: flush one-time costs (kernel compile, allocator segments, task
        init) out of the capture path on an INDEPENDENT stream, mirroring the
        exact op sequence that is about to be captured. Returns elapsed s.
        WARNING: must stay BEFORE the capture of the given tables -- eager
        compute after a capture/replay poisons the graph pool storage."""
        T = self.timesteps
        n_dims = self._n_dims
        n_samples = self._n_samples
        d = self.dynamics
        model = self.model
        mask_atoms = self._mask_atoms
        mask_residues = self._mask_residues
        _tw0w = time.perf_counter()
        if nwarm > 0:
            wm = torch.npu.Stream()
            with torch.npu.stream(wm):
                z_w = z_ref.detach().clone()
                p_w = p_ref.detach().clone()
                s_w = torch.zeros_like(self._buf['s'])
                e_w = torch.zeros_like(self._buf['e'])
                for k in range(nwarm):
                    _per = {'math': 0.0, 'mask': 0.0, 'egnn': 0.0, 'tail': 0.0}
                    _b0 = time.perf_counter()
                    s_w.fill_(float(T // 2))
                    e_w.normal_()
                    s_n = s_w / T
                    t_n = (s_w + 1.0) / T
                    gamma_s = model.gamma(s_n)
                    gamma_t = model.gamma(t_n)
                    sigma2_ts, sigma_ts, alpha_ts = \
                        model.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_w)
                    sigma_s = model.sigma(gamma_s, target_tensor=z_w)
                    sigma_t = model.sigma(gamma_t, target_tensor=z_w)
                    _per['math'] = time.perf_counter() - _b0
                    _b0 = time.perf_counter()
                    w_cat = torch.cat(
                        [z_w[:, :n_dims], p_w[:, :n_dims]], dim=0)
                    w_mask = compute_edge_mask(w_cat, be, segs)
                    _per['mask'] = time.perf_counter() - _b0
                    _b0 = time.perf_counter()
                    eps_w, _ = d(z_w, p_w, t_n, mask_atoms, mask_residues,
                                 precomputed_edges=be,
                                 precomputed_edge_types=bet,
                                 precomputed_edge_mask=w_mask)
                    _per['egnn'] = time.perf_counter() - _b0
                    _b0 = time.perf_counter()
                    mu_w = z_w / alpha_ts[mask_atoms] - \
                        (sigma2_ts / alpha_ts / sigma_t)[mask_atoms] * eps_w
                    sigma_f = sigma_ts * sigma_s / sigma_t
                    out_w = mu_w + sigma_f[mask_atoms] * e_w
                    m_w = scatter_mean(out_w[:, :n_dims], mask_atoms, dim=0,
                                       dim_size=n_samples)
                    z_w.copy_(torch.cat(
                        [out_w[:, :n_dims] - m_w[mask_atoms],
                         out_w[:, n_dims:]], dim=1))
                    p_w[:, :n_dims].copy_(
                        p_w[:, :n_dims] - m_w[mask_residues])
                    _per['tail'] = time.perf_counter() - _b0
                    if os.environ.get('DYN_TIMER'):
                        print(f"[{label} k={k}] {{math={_per['math']*1e3:.1f}ms "
                              f"mask={_per['mask']*1e3:.1f}ms "
                              f"egnn={_per['egnn']*1e3:.1f}ms "
                              f"tail={_per['tail']*1e3:.1f}ms}}", flush=True)
            torch.npu.synchronize()
        return time.perf_counter() - _tw0w

    def _capture_chain(self, be, segs, bet):
        """Capture the whole-step op chain (coefficients -> EGNN -> mu ->
        noise -> COM projection) over the given static edge tables into a new
        NPUGraph, reading/writing the shared buffers. Run once per round in
        wrap(), and again per trapezoid re-capture."""
        T = self.timesteps
        n_samples = self._n_samples
        n_dims = self._n_dims
        d = self.dynamics
        model = self.model
        mask_atoms = self._mask_atoms
        mask_residues = self._mask_residues
        z_buf = self._buf['z']
        p_buf = self._buf['p']
        s_buf = self._buf['s']
        e_buf = self._buf['e']
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            # --- diffusion physics (exact eager arithmetic) -------------
            s_n = s_buf / T
            t_n = (s_buf + 1.0) / T
            gamma_s = model.gamma(s_n)
            gamma_t = model.gamma(t_n)
            sigma2_ts, sigma_ts, alpha_ts = \
                model.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_buf)
            sigma_s = model.sigma(gamma_s, target_tensor=z_buf)
            sigma_t = model.sigma(gamma_t, target_tensor=z_buf)

            # --- EGNN on static edge tables -----------------------------
            x_cat = torch.cat([z_buf[:, :n_dims], p_buf[:, :n_dims]], dim=0)
            edge_mask = compute_edge_mask(x_cat, be, segs)
            eps_t_lig, _ = d(
                z_buf, p_buf, t_n, mask_atoms, mask_residues,
                precomputed_edges=be, precomputed_edge_types=bet,
                precomputed_edge_mask=edge_mask)

            # --- mu / noise / COM-free subspace --------------------------
            mu_lig = z_buf / alpha_ts[mask_atoms] - \
                (sigma2_ts / alpha_ts / sigma_t)[mask_atoms] * eps_t_lig
            sigma_full = sigma_ts * sigma_s / sigma_t
            out_lig = mu_lig + sigma_full[mask_atoms] * e_buf

            mean = scatter_mean(out_lig[:, :n_dims], mask_atoms, dim=0,
                                dim_size=n_samples)
            z_buf.copy_(torch.cat(
                [out_lig[:, :n_dims] - mean[mask_atoms],
                 out_lig[:, n_dims:]], dim=1))
            p_buf[:, :n_dims].copy_(
                p_buf[:, :n_dims] - mean[mask_residues])
        return g

    def _recapture(self, s_int):
        """L8 trapezoid: swap in a smaller, pruned edge table mid-round and
        re-capture the step graph when the coords have collapsed enough that
        many table rows can never be real edges again.

        Correctness model:
          * re-captured (ll/x) rows are pruned by dist <= cutoff+margin on
            the CURRENT coords -- a superset of the current real edges, and a
            (margin-guarded) superset of future ones (remaining steps move
            coords by <= O(sigma_s), printed for tuning).
          * coverage is verified AT RESET TIME against the outgoing table
            (real edge sets must coincide on the snapshot coords); a miss
            rolls back to the untouched full table.
          * the final decode always uses the wrap-time FULL table (_be0), so
            post-loop exactness is independent of all re-captures.
        Ordering is forced by two torch_npu facts: replay() after reset()
        raises, and eager compute after a LIVE capture poisons the pool. So:
        LAST replay completes -> sync -> reset (pool freed) -> eager prune /
        warm / capture -> restore buffers (the capture ran one phantom step)
        -> replays continue on the new graph."""
        if self._g is None:
            return
        _b = time.perf_counter()
        torch.npu.synchronize()          # flush the device backlog entirely
        self._g.reset()                  # dead graph -> pool returned
        self._g = None

        z_buf, p_buf = self._buf['z'], self._buf['p']
        z_ref = z_buf.clone()            # state after step s (pre-capture)
        p_ref = p_buf.clone()
        n_dims = self._n_dims

        old_total = self._be.size(1)
        _probe = None
        if os.environ.get('L8_PROBE'):
            # measurement mode: log the pruned footprint across margins at
            # every recap point (one run yields the whole trapezoid curve)
            _probe = {m: build_edge_tables_reduced(
                self.dynamics, self._mask_atoms, self._mask_residues,
                z_ref[:, :n_dims], p_ref[:, :n_dims], m)[0].size(1)
                for m in (0.25, 0.5, 1.0, 1.5, 2.0)}
        margin = float(os.environ.get('L8_MARGIN', '0.75'))
        be, segs, bet = build_edge_tables_reduced(
            self.dynamics, self._mask_atoms, self._mask_residues,
            z_ref[:, :n_dims], p_ref[:, :n_dims], margin)
        _tb_ = time.perf_counter()

        # Coverage: real (masked) rows on the snapshot coords must be the
        # same for the pruned and the outgoing table. Equal counts suffice:
        # pruning can only REMOVE rows, so pruned-real <= outgoing-real.
        x_cat = torch.cat([z_ref[:, :n_dims], p_ref[:, :n_dims]], dim=0)
        n_old_real = int(compute_edge_mask(
            x_cat, self._be, self._segs).detach().sum().item())
        n_new_real = int(compute_edge_mask(x_cat, be, segs).detach().sum().item())
        cov_ok = (n_new_real == n_old_real)
        if not cov_ok:
            # pruner miss (should be impossible): roll back to the full table
            be, segs, bet = (self._be0, self._segs0, self._bet0)
        self._set_eager_canary(be)

        _nw = int(os.environ.get('L8_WARMUP', '1'))
        _tww = self._warmup_chain(be, segs, bet, z_ref, p_ref, _nw,
                                  label=f'recap{s_int}')
        _tw0 = time.perf_counter()
        g_new = self._capture_chain(be, segs, bet)
        _tcap = time.perf_counter() - _tw0

        # the capture ran one phantom step; restore the true continue state
        z_buf.copy_(z_ref)
        p_buf.copy_(p_ref)

        self._g = g_new
        self._be, self._segs, self._bet = be, segs, bet
        self.n_captures += 1
        self.table_sizes = {'ll': segs[0][2], 'x': segs[1][2] - segs[1][1],
                            'pp': segs[2][2] - segs[2][1],
                            'total': be.size(1),
                            'n_lig': self.table_sizes.get('n_lig', 0),
                            'n_poc': self.table_sizes.get('n_poc', 0)}
        s_n = torch.full((1,), s_int / self.timesteps, device=x_cat.device)
        sigma_s = float(self.model.sigma(
            self.model.gamma(s_n), target_tensor=x_cat).item())
        if os.environ.get('DYN_TIMER'):
            print(f"[recap s={s_int}] total {old_total}->{be.size(1)} "
                  f"(ll {segs[0][2]} x {segs[1][2]-segs[1][1]} "
                  f"pp {segs[2][2]-segs[2][1]}) margin={margin} "
                  f"sigma_s={sigma_s:.3f} real {n_old_real}->{n_new_real} "
                  f"cov={'ok' if cov_ok else 'FALLBACK'} "
                  f"warmup={_tww:.3f}s capture={_tcap:.3f}s "
                  f"sync+cost={time.perf_counter()-_b:.3f}s"
                  + (f" probe={" ".join([f'{m}:{t}' for m, t in _probe.items()])}"
                     if _probe else "") + "", flush=True)

    def final_egnn(self, z, p, t, mask_atoms, mask_residues):
        """Post-loop final decode (p(x,h|z0)): the dynamics EGNN forward with
        the WRAP-TIME FULL edge tables (`_be0`, never the trapezoid-pruned
        ones), edge mask recomputed eagerly from the current coords. Avoids
        the dynamic-shape nonzero, which became pathological (~4 s) on
        graph-pool memory, and keeps post-loop exactness independent of any
        L8 re-captures: the full table contains every same-mol/cross pair,
        so the mask-selected set == the eager get_edges set at these coords.
        """
        n_dims = self.dynamics.n_dims
        x_cat = torch.cat([z[:, :n_dims], p[:, :n_dims]], dim=0)
        edge_mask = compute_edge_mask(x_cat, self._be0, self._segs0)
        if os.environ.get('DYN_TIMER'):
            # L8 measurement: real edge count at round end (s=0 coords) -- the
            # theoretical floor the trapezoid could prune toward.
            print(f"[decode] mask_real="
                  f"{int(edge_mask.detach().sum().item())}", flush=True)
        return self.dynamics(
            z, p, t, mask_atoms, mask_residues,
            precomputed_edges=self._be0, precomputed_edge_types=self._bet0,
            precomputed_edge_mask=edge_mask)

    def step(self, s_int):
        """One denoising step: fill the step-index and noise buffers, replay.
        L8: when s_int crosses a L8_RECAP point, the in-graph edge table is
        swapped for a pruned smaller one (trapezoid) and the graph re-captured.
        Returns (z_buf, p_buf); caller treats them as the new state (the
        graph rewrote them in place)."""
        _pb = time.perf_counter()
        b = self._buf
        b['s'].fill_(float(s_int))
        b['e'].normal_()
        _pf = time.perf_counter()
        self._g.replay()
        t1 = time.perf_counter()
        if os.environ.get('DYN_TIMER') or os.environ.get('PERSTEP'):
            self._te += t1 - _pf
            self._tb += _pf - _pb
            self._step_count += 1
        if os.environ.get('PERSTEP'):
            torch.npu.synchronize()
            t2 = time.perf_counter()
            self._sync += t2 - t1
            self._sync_host += t2 - _pb
        _se = os.environ.get('L6_SYNC_EVERY')
        if _se and s_int % int(_se) == 0:
            torch.npu.synchronize()
        if self._recaps and s_int == self._recaps[0]:
            self._recaps.pop(0)
            self._recapture(s_int)
        return b['z'], b['p']
