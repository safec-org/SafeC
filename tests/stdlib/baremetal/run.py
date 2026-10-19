#!/usr/bin/env python3
"""Run a linked bare-metal test image (from run.sh) in the Unicorn CPU
emulator and report the result globals bm_main.sc leaves behind.

usage: run.py <alias> <image.bin> <symbols.txt>
  image.bin   flat binary (llvm-objcopy -O binary), loaded at LOAD_ADDR
  symbols.txt `llvm-nm` output for the same ELF
Exit status: 0 if main returned with g_done set and g_fails == 0.
"""
import struct
import sys

from unicorn import (Uc, UcError, UC_ARCH_ARM, UC_ARCH_ARM64, UC_ARCH_RISCV,
                     UC_ARCH_X86, UC_MODE_32, UC_MODE_64, UC_MODE_ARM,
                     UC_MODE_THUMB, UC_MODE_MCLASS, UC_MODE_RISCV32,
                     UC_MODE_RISCV64, UC_PROT_ALL)
from unicorn import arm_const as A, arm64_const as A64, riscv_const as RV, x86_const as X

LOAD_ADDR = 0x10000
RAM_SIZE = 32 * 1024 * 1024
SENTINEL = 0x1000          # return address of main; never executed
STEP_LIMIT = 2_000_000_000

alias, image_path, syms_path = sys.argv[1:4]
image = open(image_path, 'rb').read()
syms = {}
for line in open(syms_path):
    parts = line.split()
    if len(parts) == 3:
        syms[parts[2]] = int(parts[0], 16)

thumb = alias.startswith('thumb') or alias.startswith('arm-')
if alias == 'x86':
    uc, ptr = Uc(UC_ARCH_X86, UC_MODE_32), 4
elif alias == 'x86_64':
    uc, ptr = Uc(UC_ARCH_X86, UC_MODE_64), 8
elif alias == 'aarch32':
    uc, ptr = Uc(UC_ARCH_ARM, UC_MODE_ARM), 4
elif thumb:
    uc, ptr = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS), 4
elif alias == 'aarch64':
    uc, ptr = Uc(UC_ARCH_ARM64, UC_MODE_ARM), 8
elif alias.startswith('rv32'):
    uc, ptr = Uc(UC_ARCH_RISCV, UC_MODE_RISCV32), 4
elif alias.startswith('rv64'):
    uc, ptr = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64), 8
else:
    sys.exit(f'unknown target {alias}')

uc.mem_map(0, RAM_SIZE, UC_PROT_ALL)
uc.mem_write(LOAD_ADDR, image)
sp = RAM_SIZE - 0x100
entry = syms['main']

if alias == 'x86':
    sp -= 4
    uc.mem_write(sp, struct.pack('<I', SENTINEL))
    uc.reg_write(X.UC_X86_REG_ESP, sp)
elif alias == 'x86_64':
    sp -= 8
    uc.mem_write(sp, struct.pack('<Q', SENTINEL))
    uc.reg_write(X.UC_X86_REG_RSP, sp)
elif alias == 'aarch64':
    uc.reg_write(A64.UC_ARM64_REG_SP, sp)
    uc.reg_write(A64.UC_ARM64_REG_X30, SENTINEL)
    uc.reg_write(A64.UC_ARM64_REG_CPACR_EL1, 3 << 20)   # allow FP/SIMD
elif alias.startswith('rv'):
    uc.reg_write(RV.UC_RISCV_REG_SP, sp)
    uc.reg_write(RV.UC_RISCV_REG_RA, SENTINEL)
else:  # ARM / Thumb
    uc.reg_write(A.UC_ARM_REG_SP, sp)
    uc.reg_write(A.UC_ARM_REG_LR, SENTINEL | (1 if thumb else 0))
    if thumb:
        entry |= 1

def read32(name):
    return struct.unpack('<i', uc.mem_read(syms[name], 4))[0]

# BM_TRACE=1: remember the last basic blocks executed, printed on failure.
import os, collections
trace = collections.deque(maxlen=40) if os.environ.get('BM_TRACE') else None
if trace is not None:
    from unicorn import UC_HOOK_BLOCK
    uc.hook_add(UC_HOOK_BLOCK, lambda u, addr, size, _: trace.append(addr))

err = None
try:
    uc.emu_start(entry, SENTINEL, count=STEP_LIMIT)
except UcError as e:
    err = e

pc = None
try:
    pc = {'x86': X.UC_X86_REG_EIP, 'x86_64': X.UC_X86_REG_RIP,
          'aarch64': A64.UC_ARM64_REG_PC}.get(alias)
    if pc is None:
        pc = RV.UC_RISCV_REG_PC if alias.startswith('rv') else A.UC_ARM_REG_PC
    pc = uc.reg_read(pc)
except Exception:
    pass

done, fails, checks, line = (read32('g_done'), read32('g_fails'),
                             read32('g_checks'), read32('g_first_fail_line'))
aborted = read32('g_aborted') if 'g_aborted' in syms else 0
status = 'ok' if (err is None and done == 1 and fails == 0) else 'FAILED'
detail = f'checks={checks} fails={fails}'
if fails:
    detail += f' first-fail=bm_main.sc:{line}'
if aborted:
    detail += ' (abort() called)'
if err is not None:
    detail += f' emulator: {err} at pc={pc:#x}' if pc is not None else f' emulator: {err}'
elif done != 1:
    detail += f' (main did not finish; pc={pc:#x})'
print(f'{alias:8s} {status:6s} {detail}')
if trace is not None and status != 'ok':
    addrs = sorted(syms.items(), key=lambda kv: kv[1])
    def sym(a):
        best = '?'
        for n, v in addrs:
            if v <= a: best = f'{n}+{a - v:#x}'
            else: break
        return best
    for a in trace:
        print(f'    {a:#x} {sym(a)}')
sys.exit(0 if status == 'ok' else 1)
