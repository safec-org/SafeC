// SafeC Standard Library — floating-point environment implementation
#include "fenv.h"

// fenv.h C functions (all take/return fenv_t / fexcept_t which fit in int/long)
extern int feclearexcept(int excepts);
extern int fetestexcept(int excepts);
extern int feraiseexcept(int excepts);
extern int fegetround();
extern int fesetround(int round);
extern int fegetenv(void* envp);
extern int fesetenv(const void* envp);
extern int feholdexcept(void* envp);

int fenv_clear(int excepts)       { unsafe { return feclearexcept(excepts); } }
int fenv_test(int excepts)        { unsafe { return fetestexcept(excepts); } }
int fenv_raise(int excepts)       { unsafe { return feraiseexcept(excepts); } }
int fenv_get_round()              { unsafe { return fegetround(); } }
int fenv_set_round(int mode)      { unsafe { return fesetround(mode); } }
int fenv_save(void* env)          { unsafe { return fegetenv(env); } }
int fenv_restore(const void* env) { unsafe { return fesetenv(env); } }
int fenv_save_clear(void* env)    { unsafe { return feholdexcept(env); } }
