# defer

Defer stack written in C99. Cleanup callbacks are registered with a pointer and a function, then run
in LIFO order when the stack is deinitialized.

## Usage

Compile `defer.c` alongside your project and include `defer.h`. See `Makefile` for recommended
compilation flags.

## Functions

**`defer_init`** Initialize an empty defer stack.

**`defer_deinit`** Run all deferred calls in LIFO order, then free the stack.

**`defer_push`** Push a deferred call onto the stack. Returns the pointer.

**`defer_pop`** Cancel the deferred call associated with a given pointer without running it. Scans
the stack for the first matching entry and cancels it. Returns the pointer.

## Implementation notes

- Each stack node is a heap-allocated linked list entry storing a pointer and a function
