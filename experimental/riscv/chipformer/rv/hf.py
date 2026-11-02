"""The RV32 core in a stock Qwen3_5ForCausalLM, with the snapshot injected into the HF cache.

    model, lay, ids = build_model(c, tokens)       # chip layers only (+ one zero attention layer)
    check(c, model, lay, ids, toks, mode="prefill")  # or "decode": the residual equals the IR's

The state snapshot (ROM, RAM, registers, PC: the heads' ``init``) goes into the linear-attention
layers' ``recurrent_states`` (state row = key dim, columns = value dims) after a one-token prefix
whose embedding is 0 (a text token: it leaves every conv window and state at 0). The trace then
runs from GO either as one chunked prefill (``torch_chunk_gated_delta_rule`` with that initial
state) or token by token (``torch_recurrent_gated_delta_rule``).
"""

import numpy as np
import torch

from chipformer.q35 import ir
from chipformer.q35.emit import check_bf16, embedding_rows, emit_layers, vhead_slots
from chipformer.q35.tiny import finish, host, pick_layout


def build_model(c, toks: list[int], dtype=torch.bfloat16):
    """A model whose vocabulary is the distinct tokens of ``toks`` (ids remapped)."""
    model_cls, config_cls, base, zero_gain = host(c.target)
    vocab = sorted(set(toks))
    ids = {t: i for i, t in enumerate(vocab)}
    ids[None] = len(vocab)  # the zero prefix token
    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(len(vocab) + 1) // 64) * 64
    tc = dict(base, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate)
    sd = emit_layers(c, lay, tc, 0, "model.")
    emb = torch.zeros(n_vocab, lay.hidden)
    emb[: len(vocab)] = embedding_rows(c, lay, vocab, lay.hidden)
    sd["model.embed_tokens.weight"] = emb
    norm = torch.full((lay.hidden,), zero_gain)
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm
    model = model_cls(config_cls(**tc))
    for p in model.parameters():
        p.data.zero_()
    model.load_state_dict(sd, strict=False)
    finish(model)
    return model.to(dtype).eval(), lay, ids


def inject(c, cache, layer: int | None = None) -> None:
    """Write the heads' initial states (the snapshot) into the HF cache's recurrent states
    (``layer``: only that chip layer's, into a one-layer model's cache)."""
    r = c.target.value_heads // c.target.key_heads
    for (ni, x), (L, _, vh) in vhead_slots(c, r).items():
        if layer is not None and L != layer:
            continue
        S = cache.layers[0 if layer is not None else L].recurrent_states[0][0, vh]
        S.zero_()
        init = c.nodes[ni].init or []
        for row, vec in (init[x] if x < len(init) else {}).items():
            S[row] = torch.tensor(vec, dtype=S.dtype, device=S.device)


def check(c, model, lay, ids, toks: list[int], mode: str = "prefill", tokens: bool = False):
    """Run toks (from GO) through the model with the snapshot injected first; returns the
    number of positions compared. Asserts the residual equals the IR evaluator's on every IR dim
    at every position. With ``tokens``, returns (positions, inexact dims, wrong next-token
    predictions) instead of asserting (fp32 is not bit-exact: see docs/rv32.md)."""
    want = ir.evaluate(c, toks, check=False)
    dims = list(range(1, len(c.dims)))
    cols = [lay.dim(d) for d in dims]
    dev = next(model.parameters()).device
    x = torch.tensor([[ids[t] for t in toks]], device=dev)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[ids[None]]], device=dev), use_cache=True)
        cache = out.past_key_values
        inject(c, cache)
        hs = []
        if mode == "prefill":
            o = model(input_ids=x, past_key_values=cache, use_cache=True,
                      output_hidden_states=True)
            hs.append(o.hidden_states[c.n_layers()][0])
        else:
            for t in range(len(toks)):
                o = model(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
                hs.append(o.hidden_states[c.n_layers()][0])
        h = torch.cat(hs).double().cpu().numpy()[:, cols]
    w = want[:, dims]
    bad = np.nonzero((h != w).any(0))[0]
    if tokens:
        from chipformer.rv import core

        mat = np.zeros_like(want)
        mat[:, dims] = h
        pred = core.predictions(c, mat)
        wrong = sum(pred[t] != toks[t + 1] for t in range(len(toks) - 1))
        return h.shape[0], len(bad), wrong
    assert not len(bad), [(c.dims[dims[i]].name, int(np.argmax(h[:, i] != w[:, i])),
                           float(h[np.argmax(h[:, i] != w[:, i]), i]),
                           float(w[np.argmax(h[:, i] != w[:, i]), i])) for i in bad[:5]]
    return h.shape[0]


def check_layers(c, toks: list[int], modes=("prefill",), dtype=torch.bfloat16) -> int:
    """``check`` one chip layer at a time: each layer is a one-layer stock model (plus the zero
    attention layer) fed the previous layer's residual as ``inputs_embeds``, with its heads'
    snapshot injected. The same modules and arithmetic as the whole model, in the memory of one
    layer. ``modes``: "prefill" and / or "decode". Returns the number of positions compared."""
    model_cls, config_cls, base, _ = host(c.target)
    lay = pick_layout(len(c.dims))
    tc = dict(base, hidden_size=lay.hidden, vocab_size=64, num_hidden_layers=2,
              layer_types=["linear_attention", "full_attention"],
              linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate)
    vocab = sorted(set(toks))
    emb = embedding_rows(c, lay, vocab, lay.hidden)
    xs = {m: emb[[vocab.index(t) for t in toks]][None].to(dtype) for m in modes}
    zero = torch.zeros(1, 1, lay.hidden, dtype=dtype)
    with torch.no_grad():
        for L in range(c.n_layers()):
            model = model_cls._from_config(config_cls(**tc), dtype=dtype).eval()
            for p in model.parameters():
                p.data.zero_()
            model.load_state_dict(emit_layers(c, lay, tc, -L, "model.", only=[L]), strict=False)
            for mode, x in xs.items():
                cache = model(inputs_embeds=zero, use_cache=True).past_key_values
                inject(c, cache, L)
                if mode == "prefill":
                    xs[mode] = model(inputs_embeds=x, past_key_values=cache, use_cache=True,
                                     output_hidden_states=True).hidden_states[1]
                else:
                    xs[mode] = torch.cat([model(inputs_embeds=x[:, t:t + 1], past_key_values=cache,
                                                use_cache=True, output_hidden_states=True
                                                ).hidden_states[1] for t in range(x.shape[1])], 1)
            del model, cache
    want = ir.evaluate(c, toks, check=False)
    dims = list(range(1, len(c.dims)))
    w = want[:, dims]
    for mode, x in xs.items():
        h = x[0].double().numpy()[:, [lay.dim(d) for d in dims]]
        bad = np.nonzero((h != w).any(0))[0]
        assert not len(bad), (mode, [(c.dims[dims[i]].name, int(np.argmax(h[:, i] != w[:, i])),
                                      float(h[np.argmax(h[:, i] != w[:, i]), i]),
                                      float(w[np.argmax(h[:, i] != w[:, i]), i]))
                                     for i in bad[:5]])
    return len(toks)
