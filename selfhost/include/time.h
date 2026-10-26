// time.h — playground subset (the playground has no clock: both return 0)
#pragma once
typedef long time_t;
typedef long clock_t;
#define CLOCKS_PER_SEC 1000000L
extern long time(long* t);
extern long clock();
