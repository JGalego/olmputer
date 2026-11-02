"""Turn a scheduled q35 ``Graph`` into Qwen3.5 weights.

Residual layout of the chip region, starting at ``Layout.off`` (Qwen's own dims come first in a
merged model):

    [ballast x m] [tau] [IR dims 1..n-1] [unused]

Chip tokens carry ``B`` in every ballast dim. With m * B^2 / hidden a power of 4, every RMSNorm
on a chip token divides by exactly sqrt(m / hidden) * B, so a residual value x reaches the
projections as x * c with c = sqrt(hidden / m) / B, a power of 2; weights carry 1 / c. The IR
constant ONE is the normed ballast, B * c.
"""

import math
from dataclasses import dataclass, field

import torch

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35.ir import FFN, HD, Graph, Head, K


@dataclass
class Layout:
    hidden: int
    off: int = 0  # first chip dim
    m: int = 1  # ballast dims
    log2_ballast: int = 40
    slots: dict[int, int] | None = None  # IR dim -> residual slot, when slots are reused

    @property
    def ballast(self) -> float:
        return 2.0**self.log2_ballast

    @property
    def scale(self) -> float:
        """c: normed value of a residual value 1 on a chip token."""
        mean = self.m * self.ballast**2 / self.hidden
        log4 = math.log(mean, 4)
        assert abs(log4 - round(log4)) < 1e-12, "m * B^2 / hidden must be a power of 4"
        return 2.0 ** -round(log4)

    @property
    def tau(self) -> int:
        return self.off + self.m

    def dim(self, d: int) -> int:
        """Residual index of IR dim d >= 1."""
        if self.slots is not None:
            return self.off + self.m + 1 + self.slots[d]
        return self.off + self.m + d


@dataclass
class Alloc:
    """Residual slots shared by IR dims whose lifetimes do not overlap.

    A residual write adds, so a slot is handed over by subtracting its old value: an MLP node
    defining the new dim does it with one extra unit; a head's dim is cleared by a unit in the
    MLP sublayer just before it.
    """

    slots: dict[int, int]
    n_slots: int
    subtract: dict[int, int] = field(default_factory=dict)  # FFN-defined dim -> old dim
    clears: dict[int, list[int]] = field(default_factory=dict)  # MLP level -> old dims
    final: list[int] = field(default_factory=list)  # each slot's last owner


def reads(n) -> list[int]:
    if isinstance(n, FFN):
        return [d for _, a, b in n.terms for e in (a, b) for d in e.terms]
    exprs = [c.expr for ch in (n.q, n.k) for c in ch.values()]
    for vh in n.vheads():
        exprs += [c.expr for c in vh.v.values()] + [vh.a, *vh.z.values()]
        exprs += [] if vh.beta is None else [vh.beta]
    return [d for e in exprs for d in e.terms]


def allocate(g: Graph, clearable: set[int] | None, keep: set[int], unit_cap: int) -> Alloc:
    """Assign slots; dims in ``keep`` are never handed over (outputs read by the lm_head).

    A handover clears the old value with one "a-form" unit, -x * relu(ONE), exact for any bf16
    x, so every dim is clearable unless ``clearable`` restricts it.
    """
    top = max(n.level for n in g.nodes) + 1
    last = {d: g.dims[d].level for d in range(1, len(g.dims))}
    for n in g.nodes:
        for d in reads(n):
            if d != ONE:
                last[d] = max(last[d], n.level)
    for d in keep:
        last[d] = top
    units = dict(g.units)
    slots: dict[int, int] = {}
    owner: list[int] = []  # slot -> current dim
    a = Alloc(slots, 0)

    def take(d, level, by_ffn):
        for s, o in enumerate(owner):
            if o in keep or (clearable is not None and o not in clearable):
                continue
            born = g.dims[o].level  # the old value must already be in the sublayer's input
            if by_ffn and last[o] <= level and born < level and units.get(level, 0) < unit_cap:
                units[level] = units.get(level, 0) + 1
                a.subtract[d] = o
            elif not by_ffn and last[o] <= level - 1 and born < level - 1 and level >= 3                     and units.get(level - 1, 0) < unit_cap:
                units[level - 1] = units.get(level - 1, 0) + 1
                a.clears.setdefault(level - 1, []).append(o)
            else:
                continue
            owner[s] = d
            slots[d] = s
            return
        slots[d] = len(owner)
        owner.append(d)

    for d in sorted(g.token_fns):
        take(d, 0, False)
    defined = {}
    for n in g.nodes:
        if isinstance(n, FFN):
            if not n.into:
                defined.setdefault(n.level, []).append((n.out, True))
        else:  # heads: their own output dims (not the ones they add onto)
            for vh in n.vheads():
                if not vh.into:
                    defined.setdefault(n.level, []).extend((d, False) for _, d, _ in vh.out)
    for level in sorted(defined):
        for d, by_ffn in defined[level]:
            take(d, level, by_ffn)
    a.n_slots = len(owner)
    a.final = list(owner)
    return a


def bf16_parts(x: float, n: int) -> list[float]:
    """Split x into at most n bf16 values that sum to it exactly."""
    parts = []
    for _ in range(n):
        if x == 0:
            break
        p = torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).double().item()
        parts.append(p)
        x -= p
    if x != 0:
        raise ValueError(f"constant needs more than {n} bf16 parts")
    return parts


def row(e: Expr, lay: Layout, out: torch.Tensor, w: float = 1.0) -> None:
    """Add w * e as a projection row (input side: normed residual).

    A constant rides on the ballast dims; it is split over them so that each weight is a bf16
    value (the matmul sums the parts exactly in fp32).
    """
    c = lay.scale
    for d, coef in e.terms.items():
        if d == ONE:
            for i, p in enumerate(bf16_parts(w * float(coef) / (lay.ballast * c), lay.m)):
                out[lay.off + i] += p
        else:
            out[lay.dim(d)] += w * float(coef) / c


def check_bf16(name: str, t: torch.Tensor) -> None:
    b = t.to(torch.bfloat16).to(t.dtype)
    bad = ((b != t) | ~torch.isfinite(b)).nonzero()  # (a weight beyond bf16 rounds to inf)
    if len(bad):
        i = tuple(bad[0].tolist())
        raise ValueError(f"{name}: weight {t[i].item()!r} at {i} is not a bf16 value")


def clearable_dims(g: Graph, traces: list[list[int]]) -> set[int]:
    """Dims a single unit can clear: every value seen is 0 or >= 0.15 (silu is exact there)."""
    import numpy as np

    from chipformer.q35 import ir

    ok = np.ones(len(g.dims), dtype=bool)
    for toks in traces:
        m = ir.evaluate(g, toks, check=False)
        ok &= ((m == 0) | (m >= 0.15)).all(0)
    return {d for d in range(1, len(g.dims)) if ok[d]}


def is_olmo(tc: dict) -> bool:
    """An OLMo-Hybrid config (beta = 2 sigmoid(b), separate projections, gains w)."""
    return "linear_allow_neg_eigval" in tc


def vhead_slots(g: Graph, r: int) -> dict:
    """(IR node index, value head x) -> (layer, key head, value head) in the emitted model:
    a layer's heads in node order, r value heads to a key head (a node with more value heads
    than r takes further key heads, with the same queries and keys)."""
    out, used = {}, {}
    for i, n in enumerate(g.nodes):
        if isinstance(n, Head):
            L = (n.level - 1) // 2
            j0 = used.get(L, 0)
            for x in range(len(n.vheads())):
                j = j0 + x // r
                out[i, x] = (L, j, j * r + x % r)
            used[L] = j0 + -(-len(n.vheads()) // r)
    return out


def emit_layers(g: Graph, lay: Layout, tc: dict, first: int, prefix: str,
                alloc: Alloc | None = None, keep: set[int] | None = None,
                only: list[int] | None = None) -> dict:
    """Weights of the chip layers, named ``{prefix}layers.{first + i}.*`` (``only``: just
    those layers i).

    With ``keep``, the MLP sublayer of the nodes named ``exit.*`` also clears every chip dim
    except ``keep`` (and the ballast), so nothing of the circuit is left for later layers.
    """
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc.get("linear_key_head_dim", HD), tc.get("linear_value_head_dim", HD)
    inter = tc["intermediate_size"]
    olmo = is_olmo(tc)
    # With more value heads than key heads (4B: 32 vs 16), value head i reads key head
    # i // r; a circuit head uses the first value head of its group and leaves the rest at zero.
    r = nv // nk
    assert r * nk == nv
    slots = vhead_slots(g, r)
    f32 = torch.float32
    out = {}
    exit_level = None
    if keep is not None:
        exit_level = max(n.level for n in g.nodes if n.name.startswith("exit."))
        assert all(n.level < exit_level for n in g.nodes if not n.name.startswith("exit."))
        live = alloc.final if alloc else list(range(1, len(g.dims)))
        exit_clear = [d for d in live if d not in keep]
    one = Expr({ONE: 1})
    for L in range(g.n_layers()) if only is None else only:
        p = f"{prefix}layers.{first + L}."
        qkv = torch.zeros(2 * nk * dk + nv * dv, h, dtype=f32)
        convw = torch.zeros(2 * nk * dk + nv * dv, 1, tc["linear_conv_kernel_dim"], dtype=f32)
        zw = torch.zeros(nv * dv, h, dtype=f32)
        bw = torch.zeros(nv, h, dtype=f32)
        aw = torch.zeros(nv, h, dtype=f32)
        ow = torch.zeros(h, nv * dv, dtype=f32)
        gate = torch.zeros(inter, h, dtype=f32)
        up = torch.zeros(inter, h, dtype=f32)
        down = torch.zeros(h, inter, dtype=f32)
        heads = [i for i, n in enumerate(g.nodes) if isinstance(n, Head) and n.level == 2 * L + 1]
        for i_n in heads:
            n = g.nodes[i_n]
            for x, vhd in enumerate(n.vheads()):
                _, j, vh = slots[i_n, x]
                assert j < nk, (n.name, j, nk)
                if vh % r == 0:  # the key head's queries and keys (shared by its value heads)
                    for base, chans in ((j * dk, n.q), (nk * dk + j * dk, n.k)):
                        for i, ch in chans.items():
                            row(ch.expr, lay, qkv[base + i])
                            for lag, w in enumerate(ch.taps):
                                convw[base + i, 0, -1 - lag] = w
                base = 2 * nk * dk + vh * dv
                for i, ch in vhd.v.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lag, w in enumerate(ch.taps):
                        convw[base + i, 0, -1 - lag] = w
                # beta = sigmoid(128) = 1, or a write gate: sigmoid(256 g - 128), 1 or exactly 0;
                # OLMo's 2 sigmoid(b): b = 0 (a zero row) or 256 (g - 1)
                if olmo:
                    if vhd.beta is not None:
                        row(256 * vhd.beta - 256 * one, lay, bw[vh])
                else:
                    beta = Expr({ONE: 128}) if vhd.beta is None else 256 * vhd.beta - 128 * one
                    row(beta, lay, bw[vh])
                row(vhd.a, lay, aw[vh])
                for col, e in vhd.z.items():
                    row(e, lay, zw[vh * dv + col])
                for col, d, w in vhd.out:
                    ow[lay.dim(d), vh * dv + col] += w
        one = Expr({ONE: 1})
        units = []  # (coef, a, b, output dim)
        for n in g.nodes:
            if not (isinstance(n, FFN) and n.level == 2 * L + 2):
                continue
            units += [(coef, a, b, n.out) for coef, a, b in n.terms]
            if alloc and n.out in alloc.subtract:  # hand the slot over: remove the old value
                units.append((-1, Expr({alloc.subtract[n.out]: 1}), one, n.out))
        units += [(-1, Expr({o: 1}), one, o) for o in (alloc.clears.get(2 * L + 2, [])
                                                       if alloc else [])]
        if exit_level == 2 * L + 2:
            units += [(-1, Expr({o: 1}), one, o) for o in exit_clear]
        assert len(units) <= inter, (L, len(units), inter)
        for u, (coef, a, b, d) in enumerate(units):
            row(b, lay, gate[u], K)
            row(a, lay, up[u], 1 / K)
            down[lay.dim(d), u] = float(coef)
        if exit_level == 2 * L + 2:  # the ballast goes too: -B per ballast dim
            for i in range(lay.m):
                u = len(units) + i
                row(Expr({ONE: K}), lay, gate[u])
                up[u, lay.off + i] = (lay.ballast / K) / (lay.ballast * lay.scale)
                down[lay.off + i, u] = -1.0
        if olmo:  # gains are w, not 1 + w
            kd = nk * dk
            layer = {
                "input_layernorm.weight": torch.ones(h, dtype=f32),
                "post_attention_layernorm.weight": torch.ones(h, dtype=f32),
                "linear_attn.q_proj.weight": qkv[:kd].clone(),
                "linear_attn.k_proj.weight": qkv[kd:2 * kd].clone(),
                "linear_attn.v_proj.weight": qkv[2 * kd:].clone(),
                "linear_attn.g_proj.weight": zw,
                "linear_attn.a_proj.weight": aw,
                "linear_attn.b_proj.weight": bw,
                "linear_attn.conv1d.weight": convw,
                "linear_attn.dt_bias": torch.full((nv,), -110.0, dtype=f32),
                "linear_attn.A_log": torch.zeros(nv, dtype=f32),
                "linear_attn.o_norm.weight": torch.ones(dv, dtype=f32),
                "linear_attn.o_proj.weight": ow,
            }
        else:
            layer = {
                "input_layernorm.weight": torch.zeros(h, dtype=f32),
                "post_attention_layernorm.weight": torch.zeros(h, dtype=f32),
                "linear_attn.in_proj_qkv.weight": qkv,
                "linear_attn.in_proj_z.weight": zw,
                "linear_attn.in_proj_a.weight": aw,
                "linear_attn.in_proj_b.weight": bw,
                "linear_attn.conv1d.weight": convw,
                "linear_attn.dt_bias": torch.full((nv,), -110.0, dtype=f32),
                "linear_attn.A_log": torch.zeros(nv, dtype=f32),
                "linear_attn.norm.weight": torch.ones(dv, dtype=f32),
                "linear_attn.out_proj.weight": ow,
            }
        layer |= {"mlp.gate_proj.weight": gate, "mlp.up_proj.weight": up,
                  "mlp.down_proj.weight": down}
        for k, t in layer.items():
            check_bf16(p + k, t)
            out[p + k] = t
    return out


def embedding_rows(g: Graph, lay: Layout, vocab: list[int], h: int) -> torch.Tensor:
    """Rows for the chip tokens ``vocab`` (chip token ids, in order)."""
    e = torch.zeros(len(vocab), h, dtype=torch.float32)
    e[:, lay.off : lay.off + lay.m] = lay.ballast
    for d in g.token_fns:
        e[:, lay.dim(d)] = torch.tensor([float(g.feature_value(d, t)) for t in vocab])
    check_bf16("embedding", e)
    return e
