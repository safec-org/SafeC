// QEMU netduinoplus2 (STM32F405): the Stm32F4 chip layouts against QEMU's
// models of the STM32F4 USART, general-purpose timer and SPI.
#include <std/hal/uart.h>
#include <std/hal/timer.h>
#include <std/hal/spi.h>

#define USART1  0x40011000UL
#define TIM2    0x40000000UL
#define SPI1    0x40013000UL

extern void qemu_exit(int failed);

static struct Uart<Stm32F4Usart> g_uart;

static void say(const &char s) { g_uart.write_str(s); }
static void finish(int ok) {
    say(ok ? "QEMU-TEST PASS\n" : "QEMU-TEST FAIL\n");
    qemu_exit(ok ? 0 : 1);
}

void qemu_main() {
    unsafe { g_uart = uart_init((&volatile unsigned int)USART1, 115200u, 84000000u); }
    say("uart: Stm32F4Usart ok\n");

    struct Timer<Stm32F4Tim> t;
    unsafe { t = timer_init((&volatile unsigned int)TIM2, 0u); }
    t.set_period(0xFFFFFFFFu);
    t.start();
    unsigned int c0 = t.read();
    volatile int spin = 0;
    while (spin < 200000) { spin = spin + 1; }
    unsigned int c1 = t.read();
    t.stop();
    if (c1 == c0) { say("tim: counter did not move\n"); finish(0); }
    t.clear_flag();
    t.set_period(1000u);
    // QEMU's timer model only sets UIF when the update interrupt is enabled
    // (real STM32s set it regardless); the NVIC line stays off, so enabling
    // DIER.UIE here takes no interrupt.
    unsafe { volatile_store((unsigned int*)(TIM2 + 0x0CUL), 1u); }
    t.start();
    int waited = 0;
    while (!t.flag_set() && waited < 10000000) { waited = waited + 1; }
    t.stop();
    if (!t.flag_set()) { say("tim: update flag never set\n"); finish(0); }
    t.clear_flag();
    if (t.flag_set()) { say("tim: flag did not clear\n"); finish(0); }
    say("tim: Stm32F4Tim counts, overflows and clears\n");

    struct SpiDevice<Stm32F4Spi> s;
    unsafe { s = spi_init((&volatile unsigned int)SPI1, SPI_MODE_0); }
    unsigned char got = s.transfer((unsigned char)0xA5);   // no slave attached: completes, reads 0
    say(got == (unsigned char)0 ? "spi: Stm32F4Spi transfer completes\n" : "spi: unexpected byte\n");
    finish(got == (unsigned char)0);
}
