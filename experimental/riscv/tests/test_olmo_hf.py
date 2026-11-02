"""The RV32 core in stock OlmoHybridForCausalLM modules (bf16, the torch reference kernels), one
chip layer at a time (hf.check_layers): with the snapshot injected, the residual equals the IR
evaluator's on every IR dim at every position, for a chunked prefill and token by token."""

import pytest

from chipformer.q35.ir import OLMO
from chipformer.rv import core, hf, ref
from chipformer.rv.programs import program


@pytest.fixture(scope="module")
def circ():
    return core.build(OLMO)


@pytest.mark.parametrize("name,modes", [("signext", ("prefill", "decode")),
                                        ("call", ("prefill",))])
def test_in_olmo_modules(circ, name, modes):
    p = program(name)
    core.snapshot(circ, ref.Machine.from_program(p))
    toks = ref.Machine.from_program(p).run()
    assert hf.check_layers(circ, toks, modes) == len(toks)
