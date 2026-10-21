// qemu-system-i386 -kernel (multiboot v1): 32-bit protected mode, paging off.
// Also the 64-bit stub that checks MmuX86_64's tables from long mode.
#define DATA     0x00500000          // keep in sync with qemu_x86.sc
#define ALIAS64  0x0000004000000000

    .section .text.start, "ax"
    .align 4
multiboot_header:
    .long 0x1BADB002                 // magic
    .long 0                          // flags
    .long -(0x1BADB002)              // checksum

    .code32
    .global _start
_start:
    cli
    lgdt  gdt_desc
    ljmp  $0x08, $1f
1:  mov   $0x10, %ax
    mov   %ax, %ds
    mov   %ax, %es
    mov   %ax, %ss
    mov   $__stack_top, %esp
    mov   %cr0, %eax              // SSE on: the i686 code uses SSE2 for 64-bit moves
    and   $0xFFFFFFFB, %eax       // EM = 0
    or    $0x2, %eax              // MP = 1
    mov   %eax, %cr0
    mov   %cr4, %eax
    or    $0x600, %eax            // OSFXSR | OSXMMEXCPT
    mov   %eax, %cr4
    call  qemu_main
2:  hlt
    jmp   2b

    .global qemu_outb                // (port, value)
qemu_outb:
    mov   4(%esp), %edx
    mov   8(%esp), %eax
    outb  %al, %dx
    ret

    .global qemu_paging_on           // CR0.PG (CR3 already loaded by activate())
qemu_paging_on:
    mov   %cr0, %eax
    or    $0x80000000, %eax
    mov   %eax, %cr0
    jmp   3f
3:  ret

    .global qemu_paging_off
qemu_paging_off:
    mov   %cr0, %eax
    and   $0x7fffffff, %eax
    mov   %eax, %cr0
    jmp   4f
4:  ret

    .global qemu_long_mode           // (pml4): PAE + LME/NXE + PG, then 64-bit code
qemu_long_mode:
    mov   4(%esp), %edi
    mov   %cr4, %eax
    or    $0x20, %eax                // PAE
    mov   %eax, %cr4
    mov   %edi, %cr3
    mov   $0xC0000080, %ecx          // EFER
    rdmsr
    or    $0x900, %eax               // LME | NXE
    wrmsr
    mov   %cr0, %eax
    or    $0x80000000, %eax
    mov   %eax, %cr0
    ljmp  $0x18, $long64

    .code64
long64:
    movabs $ALIAS64, %rbx
    mov   (%rbx), %eax
    cmp   $0xC0FFEE, %eax
    jne   fail64
    movl  $0x1234, 4(%rbx)
    mov   $(DATA + 4), %ecx
    mov   (%rcx), %eax
    cmp   $0x1234, %eax
    jne   fail64
    lea   msg_ok(%rip), %rsi
    call  puts64
    jmp   exit64
fail64:
    lea   msg_fail(%rip), %rsi
    call  puts64
exit64:
    mov   $0xf4, %dx                 // isa-debug-exit
    xor   %al, %al
    out   %al, %dx
5:  hlt
    jmp   5b
puts64:
    mov   $0x3f8, %dx
6:  lodsb
    test  %al, %al
    jz    7f
    out   %al, %dx
    jmp   6b
7:  ret

    .section .rodata
msg_ok:   .asciz "mmu: x86-64 long mode reads and writes through an alias above 4 GiB\nQEMU-TEST PASS\n"
msg_fail: .asciz "mmu: x86-64 alias check failed\nQEMU-TEST FAIL\n"

    .section .data
    .align 8
gdt:
    .quad 0
    .quad 0x00CF9A000000FFFF         // 0x08: 32-bit code
    .quad 0x00CF92000000FFFF         // 0x10: data
    .quad 0x00AF9A000000FFFF         // 0x18: 64-bit code
gdt_end:
gdt_desc:
    .word gdt_end - gdt - 1
    .long gdt
