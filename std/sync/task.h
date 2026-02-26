// SafeC Standard Library — Cooperative Task Scheduler
// Stackless cooperative multitasking. Tasks yield voluntarily via resume_point.
#pragma once

// Task states
#define TASK_READY    0
#define TASK_RUNNING  1
#define TASK_DONE     2

#define TASK_MAX 64

struct Task {
    void*  fn;            // task function: int(*)(void* arg, int resume_point)
    void*  arg;           // user argument
    int    state;         // TASK_READY, TASK_RUNNING, or TASK_DONE
    int    resume_point;  // where to resume (0 = start)
};

struct TaskScheduler {
    struct Task   tasks[TASK_MAX];
    int           count;     // number of registered tasks
    int           current;   // index of currently running task

    // Spawn a new task. fn signature: int fn(void* arg, int resume_point)
    //   - fn returns >0 to yield with a new resume_point
    //   - fn returns 0 to indicate completion
    // Returns task index, or -1 if scheduler is full.
    int           spawn(void* fn, void* arg);

    // Run one scheduling round (each ready task gets one turn).
    // Returns the count of still-active tasks.
    int           tick();

    // Run all tasks in round-robin until all are TASK_DONE.
    void          run_all();

    // Return number of active (non-DONE) tasks.
    int           active_count() const;
};

// Initialize a task scheduler.
struct TaskScheduler task_sched_init();
