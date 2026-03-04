#pragma once

#include <stddef.h>

typedef struct arena Arena;
typedef struct mark Mark;

Arena *arena_init(ptrdiff_t capacity, int growable);

void arena_deinit(Arena *self);

void *arena_malloc(Arena *self, int num, int size, int align) __attribute__((malloc));

void *arena_calloc(Arena *self, int num, int size, int align) __attribute__((malloc));

Mark *arena_save(Arena *self);

void arena_load(Arena *self, const Mark *mark);
