// QEMU virt, ARMv7-A (Cortex-A15): starts in SVC mode, MMU off.
    .syntax unified
    .arm
    .section .text.start, "ax"
    .global _start
_start:
    ldr   sp, =__stack_top
    ldr   r0, =vectors
    mcr   p15, 0, r0, c12, c0, 0  // VBAR
    bl    qemu_main
1:  b 1b

    .global qemu_exit             // r0 = 0 pass / 1 fail: semihosting SYS_EXIT
qemu_exit:
    cmp   r0, #0
    ldreq r1, =0x20026
    ldrne r1, =0x20023
    mov   r0, #0x18
    svc   0x123456
2:  b 2b

    .global qemu_mmu_on           // DACR all-client, TTBCR = 0 (TTBR0 only)
qemu_mmu_on:
    ldr   r1, =0x55555555
    mcr   p15, 0, r1, c3, c0, 0
    mov   r1, #0
    mcr   p15, 0, r1, c2, c0, 2
    isb
    bx    lr
    .global qemu_mmu_enable
qemu_mmu_enable:
    mrc   p15, 0, r0, c1, c0, 0
    orr   r0, r0, #1              // M
    mcr   p15, 0, r0, c1, c0, 0
    isb
    bx    lr

    .balign 32
vectors:                          // data/prefetch aborts etc.: report
    b     .
    b     abort_entry
    b     abort_entry
    b     abort_entry
    b     abort_entry
    b     .
    b     abort_entry
    b     abort_entry
abort_entry:
    ldr   sp, =__stack_top
    mrc   p15, 0, r0, c5, c0, 0   // DFSR
    mov   r1, lr
    mrc   p15, 0, r2, c6, c0, 0   // DFAR
    b     qemu_trap
