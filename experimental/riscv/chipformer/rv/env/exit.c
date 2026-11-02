// exit() for clang builds without a libc: the exit ecall (HALT).
#include "rt.h"

void exit(int code) {
    for (;;) rv_ecall2(93, code, 0, 0);
}
