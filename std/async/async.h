// SafeC Standard Library — async: everything that runs on any target
//
// Rust-style async/await as a library: futures and Poll, wakers and
// Context (future.h), goto-based stackless coroutines (coro.h), the
// single-threaded no-heap executor and tasks (executor.h), timers
// (timer.h), combinators (combinator.h), channels (channel.h), and
// Mutex/Semaphore/Notify/cancellation (sync.h). All of it is freestanding:
// bare metal and wasm32 included.
//
// Hosted extras, included separately:
//   <std/async/host.h>   block_on(&fut), HostPark (OS clock + condvar)
//   <std/async/pool.h>   ThreadPool, the multi-threaded executor
//   <std/async/net.h>    IoDriver, Runtime, AsyncTcpListener/AsyncTcpStream
#pragma once
#include <std/async/future.h>
#include <std/async/coro.h>
#include <std/async/executor.h>
#include <std/async/timer.h>
#include <std/async/combinator.h>
#include <std/async/channel.h>
#include <std/async/sync.h>
