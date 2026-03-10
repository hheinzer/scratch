#include "arena.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)

#include <sanitizer/asan_interface.h>

#define MAKE_REGION_NOACCESS(addr, size) __asan_poison_memory_region(addr, size)
#define MAKE_REGION_ADDRESSABLE(addr, size) __asan_unpoison_memory_region(addr, size)

enum { REDZONE = 64 };

#else

#define MAKE_REGION_NOACCESS(addr, size) ((void)(addr), (void)(size))
#define MAKE_REGION_ADDRESSABLE(addr, size) ((void)(addr), (void)(size))

enum { REDZONE = 0 };

#endif

enum { ALIGN = 64 };

struct arena {
    char *base;
    char *last;
    char *beg;
    char *end;
    Arena *prev;
    int growable;
};

struct mark {
    char *base;
    char *last;
    char *beg;
};

Arena *arena_init(ptrdiff_t capacity, int growable)
{
    assert(capacity > 0);

    Arena *self = malloc(sizeof(*self));
    assert(self);

    self->base = malloc(capacity);
    assert(self->base);

    MAKE_REGION_NOACCESS(self->base, capacity);

    self->last = 0;
    self->beg = self->base;
    self->end = self->base + capacity;

    self->prev = 0;
    self->growable = growable;

    return self;
}

void arena_deinit(Arena *self)
{
    while (self) {
        Arena *prev = self->prev;
        free(self->base);
        free(self);
        self = prev;
    }
}

Mark *arena_save(Arena *self)
{
    assert(self);

    char *base = self->base;
    char *last = self->last;
    char *beg = self->beg;

    Mark *mark = arena_malloc(self, 1, sizeof(*mark), sizeof(void *));
    mark->base = base;
    mark->last = last;
    mark->beg = beg;

    self->last = 0;

    return mark;
}

void arena_load(Arena *self, const Mark *mark)
{
    assert(self && mark);

    while (self->prev && self->base != mark->base) {
        Arena *prev = self->prev;
        free(self->base);
        *self = *prev;
        free(prev);
    }

    assert(self->base == mark->base && self->base <= mark->beg && mark->beg <= self->beg);
    char *last = mark->last;
    char *beg = mark->beg;

    MAKE_REGION_NOACCESS(beg, self->beg - beg);

    self->last = last;
    self->beg = beg;
}

static void grow(Arena *self, int num, int size, int align)
{
    assert(num <= (PTRDIFF_MAX - REDZONE - (align - 1)) / size);
    ptrdiff_t min_capacity = ((ptrdiff_t)num * size) + REDZONE + (align - 1);

    ptrdiff_t old_capacity = self->end - self->base;
    if (old_capacity > min_capacity) {
        min_capacity = old_capacity;
    }

    assert(min_capacity <= PTRDIFF_MAX / 2);
    ptrdiff_t capacity = 2 * min_capacity;

    Arena *next = arena_init(capacity, self->growable);

    Arena swap = *self;
    *self = *next;
    *next = swap;

    self->prev = next;
}

void *arena_malloc(Arena *self, int num, int size, int align)
{
    assert(self && num >= 0 && size > 0 && align >= 0);

    if (num == 0) {
        return 0;
    }

    if (align == 0 || (align & (align - 1)) != 0) {
        align = ALIGN;
    }

    ptrdiff_t available = self->end - self->beg;
    if (num > (available - REDZONE - (align - 1)) / size) {
        if (!self->growable) {
            abort();  // out of memory
        }
        grow(self, num, size, align);
    }

    ptrdiff_t padding = -(intptr_t)(self->beg + REDZONE) & (align - 1);
    self->last = self->beg + REDZONE + padding;

    ptrdiff_t bytes = (ptrdiff_t)num * size;
    self->beg = self->last + bytes;

    MAKE_REGION_ADDRESSABLE(self->last, bytes);

    return self->last;
}

void *arena_calloc(Arena *self, int num, int size, int align)
{
    assert(self && num >= 0 && size > 0 && align >= 0);
    void *ptr = arena_malloc(self, num, size, align);
    return ptr ? memset(ptr, 0, (ptrdiff_t)num * size) : 0;
}

void *arena_resize(Arena *self, void *last, int num, int size, int align)
{
    assert(self && num >= 0 && size > 0 && align >= 0);

    if (!last) {
        return arena_malloc(self, num, size, align);
    }

    assert(self->last == last);
    char *old_last = self->last;
    ptrdiff_t old_bytes = self->beg - old_last;

    if (num == 0) {
        MAKE_REGION_NOACCESS(old_last, old_bytes);
        self->last = 0;
        self->beg = old_last;
        return 0;
    }

    if (align == 0 || (align & (align - 1)) != 0) {
        align = ALIGN;
    }

    ptrdiff_t new_bytes = (ptrdiff_t)num * size;
    if (num <= (self->end - old_last) / size) {
        self->beg = old_last + new_bytes;
        if (old_bytes < new_bytes) {
            MAKE_REGION_ADDRESSABLE(old_last + old_bytes, new_bytes - old_bytes);
        }
        else {
            MAKE_REGION_NOACCESS(old_last + new_bytes, old_bytes - new_bytes);
        }
        return old_last;
    }

    if (!self->growable) {
        abort();
    }

    self->beg = old_last;  // reclaim
    grow(self, num, size, align);

    ptrdiff_t padding = -(intptr_t)(self->beg + REDZONE) & (align - 1);
    self->last = self->beg + REDZONE + padding;
    self->beg = self->last + new_bytes;

    MAKE_REGION_ADDRESSABLE(self->last, new_bytes);
    memmove(self->last, old_last, old_bytes < new_bytes ? old_bytes : new_bytes);
    MAKE_REGION_NOACCESS(old_last, old_bytes);

    return self->last;
}
