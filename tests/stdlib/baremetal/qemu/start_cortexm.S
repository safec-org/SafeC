// Cortex-M (QEMU netduinoplus2, STM32F405): vector table at the flash base.
    .syntax unified
    .thumb
    .section .vectors, "a"
    .word __stack_top
    .word _start                  // .thumb_func symbols already carry the Thumb bit
    .rept 14
    .word fault
    .endr

    .section .text.start, "ax"
    .global _start
    .thumb_func
_start:
    bl    qemu_main
1:  b 1b

    .thumb_func
fault:
    mov   r0, #0xFF
    b     qemu_exit

    .global qemu_exit             // r0 = 0 pass / 1 fail: semihosting SYS_EXIT
    .thumb_func
qemu_exit:
    cmp   r0, #0
    ite   eq
    ldreq r1, =0x20026
    ldrne r1, =0x20023
    mov   r0, #0x18
    bkpt  0xab
2:  b 2b
