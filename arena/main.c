#include <assert.h>
#include <stdint.h>

#include "arena.h"

static void test_basic(void)
{
    Arena *arena = arena_init(1 << 16, 0);

    // malloc: data is preserved
    int *arr = arena_malloc(arena, 4, sizeof(*arr), 0);
    assert(arr);
    for (int i = 0; i < 4; i++) {
        arr[i] = i;
    }
    for (int i = 0; i < 4; i++) {
        assert(arr[i] == i);
    }

    // calloc: memory is zeroed
    double *vec = arena_calloc(arena, 3, sizeof(*vec), 0);
    assert(vec && vec[0] == 0.0 && vec[1] == 0.0 && vec[2] == 0.0);

    arena_deinit(arena);
}

static void test_save_load(void)
{
    Arena *arena = arena_init(1 << 16, 0);

    int *arr = arena_malloc(arena, 8, sizeof(*arr), 0);
    assert(arr);
    for (int i = 0; i < 8; i++) {
        arr[i] = i;
    }

    Mark *mark = arena_save(arena);

    // allocate past the mark, then roll back
    int *tmp = arena_malloc(arena, 16, sizeof(*tmp), 0);
    assert(tmp);
    for (int i = 0; i < 16; i++) {
        tmp[i] = -1;
    }

    arena_load(arena, mark);

    // memory is reclaimed: fresh lands before tmp
    int *fresh = arena_malloc(arena, 16, sizeof(*fresh), 0);
    assert(fresh && (char *)fresh <= (char *)tmp);

    // data before the mark is intact
    for (int i = 0; i < 8; i++) {
        assert(arr[i] == i);
    }

    arena_deinit(arena);
}

static void test_malloc_zero(void)
{
    Arena *arena = arena_init(1 << 16, 0);

    // num=0 returns 0
    void *ptr = arena_malloc(arena, 0, sizeof(int), 0);
    assert(ptr == 0);

    // subsequent allocation still works
    int *arr = arena_malloc(arena, 4, sizeof(*arr), 0);
    assert(arr);

    arena_deinit(arena);
}

static void test_save_load_across_grow(void)
{
    // save before the arena grows, load must free the grown chunks
    Arena *arena = arena_init(1 << 10, 1);

    int *arr = arena_malloc(arena, 4, sizeof(*arr), 0);
    for (int i = 0; i < 4; i++) {
        arr[i] = i;
    }

    Mark *mark = arena_save(arena);

    // force a grow past the original chunk
    int *big = arena_malloc(arena, 1000, sizeof(*big), 0);
    assert(big);

    arena_load(arena, mark);

    // grown chunks freed; can allocate again from the original chunk
    int *fresh = arena_malloc(arena, 4, sizeof(*fresh), 0);
    assert(fresh);

    // data before the mark is intact
    for (int i = 0; i < 4; i++) {
        assert(arr[i] == i);
    }

    arena_deinit(arena);
}

static void test_grow(void)
{
    // start small, triggering a chunk grow mid-allocation
    Arena *arena = arena_init(64, 1);

    int *arr = arena_malloc(arena, 1000, sizeof(*arr), 0);
    for (int i = 0; i < 1000; i++) {
        arr[i] = i;
    }
    for (int i = 0; i < 1000; i++) {
        assert(arr[i] == i);
    }

    arena_deinit(arena);
}

static void test_alignment(void)
{
    Arena *arena = arena_init(1 << 16, 0);

    // interleave byte and 64-byte aligned allocations
    char *chr1 = arena_malloc(arena, 1, sizeof(*chr1), 1);
    *chr1 = 'x';

    double *dbl = arena_malloc(arena, 1, sizeof(*dbl), 64);
    *dbl = 3.14;

    char *chr2 = arena_malloc(arena, 1, sizeof(*chr2), 1);
    *chr2 = 'y';

    // values intact and dbl is correctly aligned
    assert(*chr1 == 'x' && *dbl == 3.14 && *chr2 == 'y');
    assert((uintptr_t)dbl % 64 == 0);

    arena_deinit(arena);
}

static void test_resize(void)
{
    Arena *arena = arena_init(1 << 16, 0);

    int *arr = arena_malloc(arena, 4, sizeof(*arr), 0);
    assert(arr);
    for (int i = 0; i < 4; i++) {
        arr[i] = i;
    }

    // grow in place
    arr = arena_resize(arena, arr, 8, sizeof(*arr), 0);
    assert(arr);
    for (int i = 0; i < 4; i++) {
        assert(arr[i] == i);
    }

    // shrink in place
    arr = arena_resize(arena, arr, 2, sizeof(*arr), 0);
    assert(arr[0] == 0 && arr[1] == 1);

    // resize to zero
    assert(arena_resize(arena, arr, 0, sizeof(*arr), 0) == 0);

    arena_deinit(arena);
}

static void test_resize_grow(void)
{
    // force cross-chunk resize
    Arena *arena = arena_init(64, 1);

    int *arr = arena_malloc(arena, 4, sizeof(*arr), 0);
    assert(arr);
    for (int i = 0; i < 4; i++) {
        arr[i] = i;
    }

    arr = arena_resize(arena, arr, 1000, sizeof(*arr), 0);
    assert(arr);
    for (int i = 0; i < 4; i++) {
        assert(arr[i] == i);
    }

    arena_deinit(arena);
}

int main(void)
{
    test_basic();
    test_malloc_zero();
    test_save_load();
    test_save_load_across_grow();
    test_grow();
    test_alignment();
    test_resize();
    test_resize_grow();
}
