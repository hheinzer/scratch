#pragma once

#include <stddef.h>

// Allocate `num` elements of `size` bytes aligned to `alignment` (must be a power of 2; 0 uses a
// default). Returns 0 if `num` is 0. Free with `aligned_free`.
void *aligned_malloc(int num, size_t size, size_t alignment);

// Like `aligned_malloc`, but zero-initializes the allocation.
void *aligned_calloc(int num, size_t size, size_t alignment);

// Resize a previous allocation to `num` elements of `size` bytes. If `alignment` is 0, the original
// alignment is preserved. If `ptr` is 0, behaves like `aligned_malloc`. If `num` is 0, frees `ptr`
// and returns 0.
void *aligned_realloc(void *ptr, int num, size_t size, size_t alignment);

// Free a pointer returned by `aligned_malloc`, `aligned_calloc`, or `aligned_realloc`.
void aligned_free(void *ptr);
