# defer

Defer stack written in C99. Cleanup callbacks are registered with a pointer and a function, then run
in LIFO order when the stack is deinitialized.

## Usage

Compile `defer.c` alongside your project and include `defer.h`. See `Makefile` for recommended
compilation flags.

## Functions

**`defer_init`** Initialize an empty defer stack. Returns null; the stack grows on push.

**`defer_deinit`** Run all deferred calls in LIFO order, then free the stack. Each call invokes
`func(*ptr)` using the pointer value at the time of deinit, not at push time. Safe to call with a
null stack.

**`defer_push`** Push a deferred call onto the stack. `ptr` must be passed as `&ptr` (address of the
pointer).

**`defer_pop`** Cancel the last deferred call without running it.

## Implementation notes

- Each stack node is a heap-allocated linked list entry storing a pointer-to-pointer and a function
- Storing `&ptr` rather than `ptr` means the deferred call sees the current value of the pointer at
  deinit time; setting a pointer to 0 after manually freeing it prevents a double-free
