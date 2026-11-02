"""The RV32 core compiled for OLMo-Hybrid-7B's heads (``ir.OLMO``: 30 heads of 96 x 192 a layer,
one value head per key head), against the reference trace, teacher-forced: test_rv_core.py's
tests on that target."""

import pytest
from test_rv_core import (  # noqa: F401
    test_alu_fuzz,
    test_branch_fuzz,
    test_calc,
    test_compiled,
    test_csr_stubs_and_putchar,
    test_depth_and_caps,
    test_digit_file,
    test_div_fuzz,
    test_faults,
    test_mul_fuzz,
    test_programs,
    test_refresh_after_63,
    test_snapshot_mid_run,
    test_zba_and_rodata,
)
from test_rv_fuzz import test_fuzz_ir  # noqa: F401

from chipformer.q35.ir import OLMO
from chipformer.rv import core


@pytest.fixture(scope="module")
def circ():
    return core.build(OLMO)


def test_same_depth(circ):
    assert circ.n_layers() == core.build().n_layers() == 11
