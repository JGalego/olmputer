#!/bin/bash
# Build a calc firmware with clang + lld (no gcc, no libc):  build_calc_clang.sh OUT NAME [-D..]
# e.g. the OLMo-Hybrid bridge:  build_calc_clang.sh tests/data/rv calc-olmo -DBRIDGE -DOLMO
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=$HERE/..
OUT=$(realpath "$1"); NAME=$2; shift 2
B=$(mktemp -d)
python3 "$HERE/gen_sled.py" "$B/sled.S"
# calc_in first in RAM (0x100)
sed 's/  .data : {/  .calcin : { *(.calcin) } > RAM\n  .data : {/' "$ENV/rom.ld" > "$B/calc.ld"
clang --target=riscv32-unknown-elf -march=rv32im_zba -mabi=ilp32 -O2 -ffreestanding -fno-builtin \
  -nostdlib -static -fuse-ld=lld -I"$ENV" -I"$HERE" "$@" -T"$B/calc.ld" -o "$OUT/$NAME.elf" \
  "$ENV/crt0.S" "$HERE/calc.c" "$ENV/exit.c" "$B/sled.S"
rm -rf "$B"
