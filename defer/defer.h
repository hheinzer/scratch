#pragma once

typedef struct defer Defer;

// Initialize an empty defer stack.
Defer *defer_init(void);

// Run all deferred calls in LIFO order, then free the stack.
void defer_deinit(Defer *self);

// Push a deferred call onto the stack.
void defer_push(Defer *self, void *ptr, void (*func)(void *));

// Cancel the deferred call associated with `ptr` without running it. No-op if `ptr` is not found.
void defer_pop(Defer *self, void *ptr);
