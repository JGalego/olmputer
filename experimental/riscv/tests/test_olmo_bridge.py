"""The P3 bridge for OLMo-Hybrid in the IR (``core.build(ir.OLMO, text=...)`` with OLMo's
tokenizer and calc-olmo.elf): a ``<function_calls>calc(expression=...)</function_calls>`` call is
captured, the response is OLMo's environment turn, numbers are echoed as OLMo's own digit groups
(``V(a0) BYTE(GROUP_MARK)`` records), and the circuit must predict every token it owns and own
exactly the positions the reference says."""

from pathlib import Path

import pytest
from test_rv_bridge import check

from chipformer.q35.ir import OLMO
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref

CALC = Path(__file__).parent / "data" / "rv" / "calc-olmo.elf"
TURN = "<|im_end|>\n<|im_start|>environment\n{}<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module")
def T():
    try:
        from huggingface_hub import hf_hub_download

        return BR.Text(hf_hub_download("allenai/Olmo-Hybrid-Instruct-DPO-7B", "tokenizer.json"))
    except Exception:  # noqa: BLE001 - offline
        pytest.skip("no OLMo-Hybrid tokenizer")


@pytest.fixture(scope="module")
def circuit(T):
    return core.build(OLMO, text=T)


def call(T, expr: str, quote: str = "'") -> list[int]:
    return T.encode(f"<function_calls>calc(expression={quote}{expr}{quote})</function_calls>")


def fresh_machine():
    m = ref.Machine.from_program(elf.program(str(CALC)))
    BR.park(m)
    return m


def test_depth(circuit):
    assert circuit.n_layers() == 11
    assert circuit.stats()["max_heads"] <= OLMO.key_heads


@pytest.mark.parametrize("expr,quote,want", [
    ("123456789 * 987654321", "'", "121932631112635269"),
    ("2**64-1", '"', "18446744073709551615"),
    ("10007*1", "'", "10007"),
    ("(1.2 - 3) * 4.5 - 0.75", '"', "-8.85"),
    ("-84/18", "'", "-14/3"),
    ("1/0", "'", "error"),
])
def test_calc_call(T, circuit, expr, quote, want):
    m0 = fresh_machine()
    ses = BR.Session(T, fresh_machine())
    ses.qwen(T.encode("<|im_start|>assistant\n"))
    ses.qwen(call(T, expr, quote))
    ses.qwen(T.encode("The answer is"))
    assert T.decode_bytes(ses.runs[-1]).decode() == TURN.format(want)
    check(circuit, ses, m0)


def test_digit_groups_are_olmo_tokens(T):
    """The echo of a number is OLMo's own tokenization of it."""
    ses = BR.Session(T, fresh_machine())
    ses.qwen(call(T, "2**200//3**50"))
    want = str(2**200 // 3**50)
    echo = ses.runs[-1]
    assert T.decode_bytes(echo).decode() == TURN.format(want)
    turn = T.encode(TURN.format(want))
    assert echo == turn[:-1]  # (the HALT echo is the last newline)


def test_history_then_two_calls(T, circuit):
    """A prefilled earlier call (its response as text) must not start a run; two real calls
    follow, the second after the buffer is cleared at <function_calls>."""
    m0 = fresh_machine()
    ses = BR.Session(T, fresh_machine())
    hist = call(T, "1+1") + T.encode(TURN.format("2") + "It is 2. ")
    k = len(ses.toks)
    ses.history(hist)
    skip = {k + hist.index(T.TCC)}
    ses.qwen(call(T, "7*6"))
    ses.qwen(T.encode(" and "))
    ses.qwen(call(T, "999999999999*999999999999", '"'))
    outs = [T.decode_bytes(r).decode() for r in ses.runs]
    assert outs == [TURN.format("42"), TURN.format(str(999999999999**2))]
    check(circuit, ses, m0, skip)
