// QEMU virt, AArch64: starts at EL1 with the MMU off.
    .section .text.start, "ax"
    .global _start
_start:
    ldr   x0, =__stack_top
    mov   sp, x0
    mov   x0, #(3 << 20)          // CPACR_EL1.FPEN: compiled code may use FP/SIMD
    msr   cpacr_el1, x0
    isb
    adr   x0, vectors
    msr   vbar_el1, x0
    isb
    bl    qemu_main
1:  b 1b

    .global qemu_exit             // w0 = 0 pass / 1 fail: semihosting SYS_EXIT
qemu_exit:
    ldr   x1, =0x20026            // ADP_Stopped_ApplicationExit
    cmp   w0, #0
    b.eq  2f
    ldr   x1, =0x20023            // ADP_Stopped_InternalError
2:  sub   sp, sp, #16
    str   x1, [sp]
    str   xzr, [sp, #8]
    mov   x1, sp
    mov   w0, #0x18
    hlt   #0xf000
3:  b 3b

    .global qemu_mmu_on           // x0 = root: MAIR, TCR, TTBR0, then SCTLR.M|C|I
qemu_mmu_on:
    ldr   x1, =0x04ff             // Attr0 normal WB, Attr1 device-nGnRE
    msr   mair_el1, x1
    ldr   x1, =((16) | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (2 << 32))
    msr   tcr_el1, x1             // T0SZ=16, WBWA, inner shareable, 4K, no TTBR1, 40-bit PA
    isb
    ret
    .global qemu_mmu_enable
qemu_mmu_enable:
    mrs   x0, sctlr_el1
    orr   x0, x0, #1              // M
    orr   x0, x0, #(1 << 2)       // C
    orr   x0, x0, #(1 << 12)      // I
    msr   sctlr_el1, x0
    isb
    ret

    .balign 2048
vectors:                          // every exception: report it
    .rept 16
    mrs   x0, esr_el1
    mrs   x1, elr_el1
    mrs   x2, far_el1
    b     qemu_trap
    .balign 128
    .endr
