# Olmputer: the RV32 computer inside OLMo-Hybrid-7B

`main` of [JGalego/olmputer](https://github.com/JGalego/olmputer), a fork of Qwenputer at
`bec1a0c`. The P3 machine of `docs/rv32.md` compiled into
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

### What changed (`9cc9cbf`)
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

## Phase 2: the merge (`rv/merge.py`)

- **Layout.** The chip layers first (the circuit of `core.build(ir.OLMO, text=...)`), then
  OLMo's 32 layers zero-padded from hidden 3840 to **5376** (`pick_hidden`: 21 ballast dims,
  1,327 circuit slots). The vocabulary is OLMo's 100,352 rows, then the 230,324 chip tokens.
- **Norms.** OLMo's gains are `w`, so every norm over the full residual is rescaled as
  `w * sqrt(3840 / 5376)`: the linear-attention layers' `input_layernorm` and
  `post_attention_layernorm`, the full-attention layers' `post_attention_layernorm` and
  `post_feedforward_layernorm` (post-norm: they see sublayer outputs, which are 0 on the chip
  dims), and the final norm.
- **Head.** Untied: the embedding's chip columns, with OLMo's own rows on its dims. The rows
  that pad the vocabulary to a multiple of 64 get tau = -1 like the chip tokens. As zero rows
  they scored exactly 0, above OLMo's best logit whenever all of OLMo's logits are negative:
  plain `generate` then emitted them (greedy agreement with base fell to 1/16 prompts; base's
  own sdpa vs eager agree on 11/16). The KL check slices the text vocabulary, so it missed this.
- **Config.** `head_dim` is written explicitly (128): OLMo's config leaves it to
  `hidden_size / num_attention_heads`, which is 179 at the new width.
- **Not muted.** OLMo's full-attention layers have no output gate, so chip tokens can't be
  silenced there; decoding must keep chip tokens out of OLMo's layers (`RvChat`). Muting the
  linear-attention layers alone would not make plain `generate()` work, so it is not done.
- **Build.** About 70 s on `cpu-xl`; the checkpoint is about 32 GB in bf16.

## Phase 3: bridge, firmware, driver

### Does OLMo read the byte echo? (`olmputer.py digits`, base OLMo, n = 40 products)
OLMo was asked `What is a * b?` with the calc tool, and its call was answered twice: with the
result tokenized as OLMo tokenizes it, and one byte per token as the P3 machine echoes it.

| | |
|---|---|
| OLMo calls `calc` in its own syntax | 40/40 (29 with single quotes, 11 with double) |
| exact value in the answer, OLMo's digit groups | 33/40 |
| exact value in the answer, one digit per token | **0/40** |

The 7 misses with digit groups are OLMo re-typing the number with commas and slipping a digit.
One digit per token, OLMo garbles every number. So numbers are echoed as OLMo's own tokens.

### Digit-group echo
- OLMo splits each run of digits into groups of 3 from the left (`1234567` is `123|456|7`);
  every `000`..`999` and `00`..`99` is a token.
- The P3 echo is one `BYTE` record per putchar with a 9-bit payload (260 used). Widening the
  family would shift every chip-token id after it and break the Qwen path and the published
  Qwenputer checkpoint. Instead `putchar(a0)` with `512 <= a0 < 1616` is the two-token record
  `V(a0) BYTE(511)` (`spec.GROUP_BASE`, `GROUP_MARK`); its echo is printable index `a0`:
  `512 + g` the group `g` ("000".."999"), `1512 + g` the group "00".."99".
- In the circuit, only when the bridge has digit groups (`Text.groups`): two output cases
  (`V(a0)` at the record's start, `BYTE(511)` at its second token), a one-row head that hands
  token t-1's `V` payload to token t, and the echo case using it after `BYTE(511)`. The
  standalone and Qwen circuits are unchanged. 11 layers, at most 30 heads a sublayer; 1,612
  printable tokens.

### Firmware (`env/calc/calc.c -DBRIDGE -DOLMO`, `calc-olmo.elf`)
- Parses `calc(expression="...")` and `calc(expression='...')` (backslash escapes).
- Prints OLMo's environment turn: `<|im_end|>\n<|im_start|>environment\n` VALUE
  `<|im_end|>\n<|im_start|>assistant`, and HALT's echo is the last `\n`. The specials are
  `putchar(256 + k)`: `<|im_end|>`, `<|im_start|>`, `environment`, `assistant`.
- Every run of digits (integer part, fraction digits, a fraction's numerator and denominator)
  is printed as OLMo's groups; no newline after the value.
- Built with clang 14 + lld and no libc (`build_calc_clang.sh`; `env/exit.c` is the only libc
  function calc needs). `.insn` works in clang 14.

### Driver (`rv/chat.py`)
- `RvChat` loads either host. On OLMo: the HF path only (Qwen's fast and executor chip paths
  are Qwen3.5 module code), no position embeddings (NoPE), eos {`<|endoftext|>`,
  `<|im_end|>`}, and eos ends the loop only where OLMo, not the circuit, predicted it
  (`<|im_end|>` is also an echo).
- `RvChat.force(ids, stream, owned)`: a machine run teacher-forced through the merged weights,
  transparent to chip tokens (Gate 2b).

## Gates 1-3 (HF Jobs; `olmputer.py` in hf-jgalego `beyond/olmputer/`)

### Gate 1: olmo-tiny (`check`)
olmo-tiny: 11 chip layers, hidden 2,816, 3,130M parameters. For each program the residual
before the exit layer equals the IR evaluator's on every circuit dim at every position (bf16,
traces up to 60k tokens, prefilled in 8,192-token chunks), and free-running greedy `generate`
reproduces the reference trace token for token.

| Run | Kernels | Programs | Exact | Trace tokens |
|---|---|---|---|---|
| LOOP, MEMCPY, rv32ui (38), rv32um (8), rv32uzba (3), c-crc32p, calc | torch fallback, A10G | 55 | 55 | 26,677 |
| the same | fla, A10G | 55 | 55 | 26,677 |
| c-sieve, c-gmpmul | torch fallback, A10G | 2 | 2 | 66,825 |
| c-qsort (generate only) | torch fallback, A10G | 1 | 1 | 240,175 |

Also exact: `loop` on the CPU fallback (cpu-upgrade). Two fixes on the way: the residual check
computed all positions' logits (19 GB at 44k tokens; now one) and prefilled 44k tokens at once
(now 8,192 at a time, the cache carrying the state).

### Gate 2a: text parity (`parity`, A100)
KL(base || merged), mean over 16 held-out prompts: **5.87e-4**; the noise floor, base sdpa vs
base eager: **5.66e-4**. Greedy 32-token continuations identical to base: **12/16**; base sdpa vs
base eager: 11/16. Teacher-forced top-1 agreement 98.5%. Tokenizer identical on 4/4 samples.

### Gate 2b: the machine in the merged weights (`force`, A100)
`RvChat.force`: a machine run teacher-forced through the merged model, chip tokens through the
chip layers only and echoed text through every layer. calc 651/651, c-strops 1,074/1,074,
rv32ui-lb 193/193, rv32um-div 142/142 circuit-owned positions exact (2,060/2,060).

### Gate 3: conversations (`chat`, A100, `calc-olmo.elf`)
24 conversations (the brief's question, then seeded products, sums, quotients and exact
fractions), each locked to the reference as it is generated and replayed through the IR:
**24/24 clean**: 0 lock errors, 0 wrong predictions and 0 owner mismatches over 191,100
circuit-owned positions, 0 stock-head mismatches. 190,709 chip tokens, 1,521 text tokens, about
157 s a conversation on the HF path.

"Use calc: what is 123456789 * 987654321?" returns "The result of 123456789 multiplied by
987654321 is 121932631112635269."

How OLMo handles the echo: OLMo's final answer holds the machine's value in 19/24 (thousands
separators ignored; 4/24 verbatim, OLMo likes commas). The 5 misses are OLMo re-typing the value
with commas and dropping or changing digits, e.g. `8441394907539` answered as
`8,441,399,007,539`. The machine's echo was exact in all 24.

### Tests
The suite on `cfaef76` (Qwen and OLMo tests, the Qwen3.5 tokenizer, 8 workers): 347 passed, 24
skipped, 2 failed: `test_fuzz_free_running_cpu`, which fails on upstream `bec1a0c` too
(`fastexec.py` does not know the bridge's `Q` family: upstream's baseline is 247 passed, 1
failed), and `test_digit_groups_are_olmo_tokens`, a test bug (fixed in `9b7d74b`, passing). The
Qwen path is byte-identical to upstream: 453/453 hashes of Qwen rv-tiny's tensors and snapshot
rows match.

## The executor on OLMo's heads (`rv/tritonexec.py`)
A head's state is a power-of-2 tile (OLMo: 96 x 192 padded to 128 x 256), the ballast columns
come from a mask and the norm divides by the real value width; Qwen3.5's tiles stay 128 x 128.
`execcheck` (A10G): the executor equals the IR on every dim at every position on the OLMo bridge
circuit (20,786 positions of four calc calls), and on c-crc32p on both OLMo's and Qwen3.5's
heads (6,283 each). Gate 3 on the executor: the same 24 conversations, token for token
identical to the HF path and clean, in 157 s instead of 3,762 s (24x).

## Phase 4: arithmetic (`evalarith`, n = 40 a set, greedy, exact rational match)
rv1 is Olmputer (the chip layers on the executor, every circuit token checked against the
reference as generated); pytool is base OLMo whose calc calls the same firmware answers on a
reference machine (its responses are byte for byte rv1's); base_nudge is base OLMo without tools.
All on fla kernels.

| Set | rv1 | pytool | base_nudge |
|---|---|---|---|
| arith (seeded synthetic) | 40/40 | 40/40 | 25/40 |
| dmx_mul_big | 40/40 | 40/40 | 14/40 |
| dmx_div_big | 37/40 | 37/40 | 11/40 |
| dmx_mul_div_multiple_longer | 24/40 | 22/40 | 16/40 |
| dmx_add_or_sub_big | 38/40 | 38/40 | 21/40 |
| total | **179/200** | 177/200 | 87/200 |

- rv1: 0 lock errors and 0 head mismatches over 299 calls and 1,481,063 chip tokens.
- rv1 and pytool write the same transcript on 184 of 200 problems; the merged model's small
  numeric differences send the other 16 down different paths, and 4 of those score differently
  (3 for rv1, 1 for pytool).
- The misses are OLMo's: wrong expressions (`dmx_mul_div_multiple_longer-1020`: 638 for 319),
  and calls that exceed the firmware's 7 bignum registers ("error (too complex: split it)",
  upstream's limit too).
