#pragma once

typedef struct defer Defer;

// Initialize an empty defer stack.
Defer *defer_init(void);

// Run all deferred calls in LIFO order, then free the stack. Each call invokes `func(*ptr)` using
// the pointer value at the time of deinit, not at push time. Safe to call with a null stack.
void defer_deinit(Defer *self);

// Push a deferred call onto the stack. `ptr` must be passed as `&ptr` (address of the pointer).
void defer_push(Defer **self, void *ptr, void (*func)(void *));

// Cancel the last deferred call without running it.
void defer_pop(Defer **self);
