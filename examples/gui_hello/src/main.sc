// gui_hello — opens a window with std/gui, draws into it for about a second
// and closes it again, so 'safeguard bundle' has a GUI program to package
// (and the bundle smoke test one that exits by itself).
#include <std/gui/gui.h>
#include <std/gui/gui_draw.h>
#include <std/thread.h>

extern int printf(const char* fmt, ...);

int main() {
    struct GuiWindow win = GuiWindow::create("Gui Hello", 320, 200);
    if (win.platform.is_null()) {
        // No display to open a window on (a headless CI runner, say).
        unsafe { printf("gui_hello: no window (no display?)\n"); }
        return 0;
    }
    struct GuiEvent ev;
    int frame = 0;
    while (frame < 60 && win.shouldClose == 0) {
        while (win.poll_event(&ev)) { }
        unsigned char shade = (unsigned char)(40 + frame * 2);
        win.fill_rect(0, 0, win.width, win.height, GuiColor::rgb((unsigned char)20, shade, (unsigned char)160));
        win.draw_text(24, 88, "Hello from SafeC", GuiColor::rgb((unsigned char)255, (unsigned char)255, (unsigned char)255), 2);
        win.present();
        thread_sleep_ms(16UL);
        frame = frame + 1;
    }
    win.destroy();
    unsafe { printf("gui_hello: window closed after %d frames\n", frame); }
    return 0;
}
