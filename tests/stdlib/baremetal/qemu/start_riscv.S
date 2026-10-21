// RISC-V virt: M-mode entry (-bios none loads us at 0x80000000).
#if __riscv_xlen == 64
#define REG_S sd
#else
#define REG_S sw
#endif
    .section .text.start, "ax"
    .global _start
_start:
    la    sp, __stack_top
    la    t0, trap_entry
    csrw  mtvec, t0
    call  qemu_main
1:  j 1b

    .align 4
trap_entry:                    // any trap: report it from M-mode (no translation)
    csrr  a0, mcause
    csrr  a1, mepc
    csrr  a2, mtval
    la    sp, __stack_top
    call  qemu_trap
2:  j 2b

    .global qemu_setup_pmp     // let S-mode reach all of memory
qemu_setup_pmp:
    li    t0, -1
    csrw  pmpaddr0, t0
    li    t0, 0x1f             // NAPOT, R W X
    csrw  pmpcfg0, t0
    ret

    .global qemu_enter_s       // a0 = S-mode entry point; does not return
qemu_enter_s:
    li    t0, 0x1800
    csrc  mstatus, t0          // MPP = 0
    li    t0, 0x0800
    csrs  mstatus, t0          // MPP = S
    csrw  mepc, a0
    mret
