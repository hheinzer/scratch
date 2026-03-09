#pragma once

#include <stddef.h>

typedef struct arena Arena;
typedef struct mark Mark;

// Allocate and initialize a new arena with the given `capacity` in bytes. If `growable`, the arena
// allocates new chunks automatically when full; otherwise, running out of memory calls `abort`.
Arena *arena_init(ptrdiff_t capacity, int growable);

// Free all memory owned by the arena, including any grown chunks.
void arena_deinit(Arena *self);

// Save the current arena position and return an opaque mark. The mark is allocated inside the
// arena. Any marks saved after this one are invalidated by a call to `arena_load`.
Mark *arena_save(Arena *self);

// Restore the arena to the position recorded in `mark`, freeing all allocations made after the
// save. Chunks allocated after the save are freed.
void arena_load(Arena *self, const Mark *mark);

// Allocate `num` elements of `size` bytes each. Returns 0 if `num` is 0. `align` must be a power of
// two, or 0 for default alignment. Calls `abort` on out-of-memory if the arena is not growable.
void *arena_malloc(Arena *self, int num, int size, int align) __attribute__((malloc));

// Same as `arena_malloc`, but zero-initializes the returned memory.
void *arena_calloc(Arena *self, int num, int size, int align) __attribute__((malloc));

// Resize the last allocation to `num` elements of `size` bytes each. Grows or shrinks in place if
// there is room; otherwise allocates a new block in a grown chunk and copies the data. If `last` is
// 0, behaves like `arena_malloc`. Returns 0 if `num` is 0. Calls `abort` on out-of-memory if the
// arena is not growable.
void *arena_resize(Arena *self, void *last, int num, int size, int align);
