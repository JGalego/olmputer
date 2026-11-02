# Olmputer: the RV32 computer inside OLMo-Hybrid-7B

Branch `olmputer` (on `main` at `bec1a0c`). The P3 machine of `docs/rv32.md` compiled into
[`allenai/Olmo-Hybrid-Instruct-DPO-7B`](https://huggingface.co/allenai/Olmo-Hybrid-Instruct-DPO-7B)
instead of Qwen3.5-4B. Design decisions and measurements go here as they are made.

## Phase 0: setup

### Upstream tests (workstation: WSL, CPU only, about 15 GB of RAM)
`uv sync --extra hf && uv run pytest tests -q` on `bec1a0c`: see "Test baseline" below.

### OLMo-Hybrid, checked against transformers 5.19.0
transformers 5.19.0 ships `olmo_hybrid` (`OlmoHybridForCausalLM`). The config and tokenizer
(`hf download` of the config and tokenizer files only) match the brief, with one addition:

- `rope_parameters.rope_theta` is `null`: the full-attention layers have no position
  encoding (NoPE).

Facts the compiler relies on, from `modeling_olmo_hybrid.py`:

| | Qwen3.5-4B | OLMo-Hybrid-7B |
|---|---|---|
| GDN heads per layer | 16 key x 32 value | 30 key x 30 value |
| key dim x value dim | 128 x 128 | 96 x 192 |
| projections | fused `in_proj_qkv`, `in_proj_z`, `in_proj_a/b` | `q_proj`, `k_proj`, `v_proj`, `g_proj`, `a_proj`, `b_proj` |
| conv | over q\|k\|v, 4 taps, no bias, silu | the same |
| beta | sigmoid(b) | 2 sigmoid(b) (`allow_neg_eigval`), computed in bf16 |
| decay | -exp(A_log) softplus(a + dt_bias) | the same |
| query scale in the kernels | 128^-0.5 = 2^-3.5 | 96^-0.5 |
| output norm | gated RMSNorm, eps 1e-6, gain `w` | gated RMSNorm, eps 1e-5, gain `w` |
| decoder norms | pre-norm, gain `1 + w` | GDN layers pre-norm, attention layers post-norm; gain `w` |
| MLP | SwiGLU 9216 | SwiGLU 11008 |
| embeddings | tied | untied |

Tool syntax: `<function_calls>calc(expression="…")</function_calls>`, then
`<|im_end|>\n<|im_start|>environment\n…<|im_end|>\n<|im_start|>assistant\n`. Every token
listed in the brief is a single token with the stated id. `<think>` is not a token. Numbers split
into groups of up to three digits from the left (`123456789` is `123|456|789`); all 1,000
three-digit tokens exist. `olmputer.py facts` (in hf-jgalego `beyond/olmputer/`) re-checks all of
this.

## Exact reads on OLMo's heads

A read is `o = sum_s qhat_s S[s]` with `qhat = l2norm(q) * dk^-0.5` in fp32, cast to bf16, then
the gated RMSNorm over the value dim. Qwen's circuit relies on two powers of two that OLMo's
shapes break. Both were measured on the transformers torch kernels (recurrent and chunk, CPU):

- **Ballast columns.** A row's ballast sits in `n_ref` columns; the norm divides by
  `A_REF * sqrt(n_ref / dv)`. Qwen: 2 of 128 gives 1/8. On OLMo 2 of 192 gives
  `sqrt(96)`, and a set bit reads back as 0.0095825 instead of 0.0095683: **not exact**.
  **3 ballast columns** of 192 give 1/8 again, so every normalized value, and every output
  weight downstream, is the same as on Qwen: measured exact.
- **Byte tables (ROM, DROM).** Bit rows only need every nonzero entry of a row to be the
  same scale times a power of two, so any query scale works. Byte rows (0..255) need the scale
  itself to be a power of two. Qwen gets 2^-4 from a 2-hot query (`2^-0.5 * 2^-3.5`). On OLMo
  no n-hot query does: `(96 n)^-0.5` is never a power of two. A 1-hot read of byte rows is
  **not exact** (measured).
  - **Fix: pad the query.** The selected row gets weight 3 and four rows that are never
    written get 3, 2, 1, 1. Then `||q||^2 = 24`, `qhat = 3 / sqrt(24 * 96) = 1/16` exactly:
    measured exact on both kernels. Four is the minimum (the pads' squares must sum to 15,
    which needs four squares).
  - All query values are multiples of K = 128, so silu stays exact.
  - fla keeps q in bf16, where `3/sqrt(24)` rounds by 1.5e-3, under the 2^-9 that a byte read
    tolerates. To be checked on GPU (Phase 1 jobs).
- **beta.** OLMo computes `2 sigmoid(b)` in bf16: exactly 1 at `b = 0` and exactly 0 at
  `b <= -256`, measured. The write gate is emitted as `b = 256 (g - 1)`; "always write" is
  `b = 0`, a zero row.
- **Decay.** `a in {0, 110, 238}` with `dt_bias = -110`, `A_log = 0` gives exactly 1, 0.5, 0
  (measured), so `DECAY` carries over.

## Capacity

Every table keeps its Qwenputer size. With one value head per key head, a Qwen key head with two
value heads becomes two OLMo heads with the same queries and keys. A row's address is matched by
an AND over all its address bits (`andpre`), so any address-to-(head, row) map works: rows
need not split on bit boundaries.

| Table | Size | Qwenputer (Qwen3.5-4B) | Olmputer (OLMo-Hybrid-7B) |
|---|---|---|---|
| ROM | 256 KB (65,536 words) | byte rows, 2-hot row pairs, 8 words a row; 1,024 words per value head; 64 value heads in 2 sublayers (L3-4) | byte rows, 1-hot + 4 pad rows, 32 words a row (128 byte columns); 92 rows a head, 2,944 words; **23 heads, 1 sublayer** |
| DROM | 128 KB (32,768 words) | as the ROM; 32 value heads, 1 sublayer (L9) | as the ROM; **12 heads** |
| RAM | 32 KB (8,192 words) | 1 word a row (two 32-column blocks: SW and SB/SH data), 128 rows, value head by address bit 6; 64 value heads in 2 sublayers (L7-8) | 1 word a row, 96 rows, bit 6 in the row address; **86 heads, 3 sublayers** |
| registers | 31 x 32 bits, 3 read ports | 3 key heads, 32 rows, 97 columns | 3 heads, the same rows |
| quotient | 8 digits x 8 alignments | 4 key x 2 value heads, 1 row each | 8 heads |
| digit file | 8 registers x 128 digits, 2 read ports | per port 4 key x 2 value heads, 128 rows | port 1 (reads rows 0-63 only): 8 heads of 64 rows; port 2: 16 heads (rows 0-63 and 64-127) |
| text buffer | 80 rows x 16 bytes | 1 key x 2 value heads (8 bytes each), 81 rows | 2 heads |
| pc, counters, run, arm | | 1 key head each, 1-2 rows | the same |

Where 192 columns help: ROM and DROM rows hold 32 words instead of 8 (the 2-hot pair trick
wasted half the columns), so the ROM drops from 2 sublayers to 1. RAM can't use them: a store
overwrites a whole row, so a row holds one word.

Totals: Qwenputer uses 191 value heads (plus 4 for the bridge) in 11 layers; Olmputer needs about
165 heads at 30 a sublayer.

## Phase 1: the circuit on OLMo's heads

### What changed (branch `olmputer`)
- `q35/ir.py`: `Target` carries `dk`, `dv`, the ballast columns, the output norm's eps and the
  host family; `ir.OLMO` is OLMo-Hybrid-7B's. A head with more value heads than the host has per
  key head takes several key heads (same queries and keys). Pad rows (`Head.pads`): never
  written, excluded from the multi-hot check; a byte head's read scale must be a power of 2
  (checked). `kernel_check` uses the target's `dk` and eps. `target_of(config)`.
- `q35/emit.py`: OLMo's weight names, gains `w` instead of `1 + w`, beta `256 (g - 1)` or a
  zero row, value heads spread over key heads (`vhead_slots`); `only=` emits chosen layers.
- `rv/core.py`: packed byte tables (`byte_table`: 1-hot + pads, 32 words a row), packed RAM
  (one word a row, 96 rows, every store token writes), the digit file's port 2 in two heads,
  snapshots sized by the target. Qwen3.5-4B keeps its layouts (`core.pairs`).
- `q35/tiny.py` (`host`, `OLMO_BASE`), `rv/tiny.py`, `rv/hf.py`, `rv/firmware.py`,
  `rv/tinycheck.py --olmo`: build and run either host's stock model.
- `rv/hf.py check_layers`: the HF residual check one chip layer at a time (a one-layer stock
  model fed the previous layer's residual), so the OLMo-shaped check fits the workstation.

### Layer plan (measured, `core.build(ir.OLMO)`)
**11 layers**, the same as Qwenputer; 144 IR heads, 160 OLMo heads, at most 7,876 MLP units a
sublayer (of 11,008).

| Layer | Heads (of 30) |
|---|---|
| L1 | pc, count, j, vcnt (4) |
| L3 | ROM 23, quotient 6 |
| L4 | registers 3, quotient 2 |
| L5 | digit file 24 |
| L6 | RAM 30 |
| L7 | RAM 30 |
| L8 | RAM 26, DROM 4 |
| L9 | DROM 8 |

L2, L10 and L11 have no heads (MLP only), as on Qwen. The ROM fitting one sublayer pays for the
RAM's third.
