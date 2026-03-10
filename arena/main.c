#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arena.h"

static void dump(const Arena *arena, const char *func)
{
    char fname[128];
    sprintf(fname, "%s.out", func);
    arena_dump(arena, fname);
}

static void test_basic(void)
{
    Arena *arena = arena_init(512, 0);

    // malloc: data is preserved
    char *hello = arena_malloc(arena, 16, sizeof(*hello), 1);
    assert(hello);
    strcpy(hello, "hello");
    assert(hello[0] == 'h' && hello[4] == 'o');

    // calloc: memory is zeroed
    char *zero = arena_calloc(arena, 16, sizeof(*zero), 1);
    assert(zero && zero[0] == '\0' && zero[4] == '\0');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_save_load(void)
{
    Arena *arena = arena_init(512, 0);

    char *perm = arena_malloc(arena, 16, sizeof(*perm), 1);
    assert(perm);
    strcpy(perm, "perm");

    Mark *mark = arena_save(arena);

    // allocate past the mark, then roll back
    char *temp = arena_malloc(arena, 16, sizeof(*temp), 1);
    assert(temp);
    strcpy(temp, "temp");

    arena_load(arena, mark);

    // memory is reclaimed: fresh lands at or before tmp
    char *fresh = arena_malloc(arena, 16, sizeof(*fresh), 1);
    assert(fresh && (char *)fresh <= (char *)temp);
    strcpy(fresh, "fresh");

    // data before the mark is intact
    assert(perm[0] == 'p' && perm[3] == 'm');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_malloc_zero(void)
{
    Arena *arena = arena_init(256, 0);

    // num=0 returns 0
    char *ptr = arena_malloc(arena, 0, sizeof(*ptr), 1);
    assert(ptr == 0);

    // subsequent allocation still works
    char *str = arena_malloc(arena, 16, sizeof(*str), 1);
    assert(str);
    strcpy(str, "ok");
    assert(str[0] == 'o');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_save_load_across_grow(void)
{
    // save before the arena grows, load must free the grown chunks
    Arena *arena = arena_init(512, 1);

    char *perm = arena_malloc(arena, 16, sizeof(*perm), 1);
    assert(perm);
    strcpy(perm, "perm");

    Mark *mark = arena_save(arena);

    // force a grow past the original chunk
    char *big = arena_malloc(arena, 400, sizeof(*big), 1);
    assert(big);

    arena_load(arena, mark);

    // grown chunks freed; can allocate again from the original chunk
    char *fresh = arena_malloc(arena, 16, sizeof(*fresh), 1);
    assert(fresh);
    strcpy(fresh, "fresh");

    // data before the mark is intact
    assert(perm[0] == 'p' && perm[3] == 'm');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_grow(void)
{
    // start small, triggering a chunk grow mid-allocation
    Arena *arena = arena_init(64, 1);

    char *fox = arena_malloc(arena, 64, sizeof(*fox), 1);
    strcpy(fox, "the quick brown fox jumps over the lazy dog");
    assert(fox[4] == 'q' && fox[40] == 'd');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_alignment(void)
{
    Arena *arena = arena_init(512, 0);

    // interleave byte and 64-byte aligned allocations
    char *chr1 = arena_malloc(arena, 1, sizeof(*chr1), 1);
    *chr1 = 'x';

    char *aligned = arena_malloc(arena, 16, sizeof(*aligned), 64);
    strcpy(aligned, "aligned");

    char *chr2 = arena_malloc(arena, 1, sizeof(*chr2), 1);
    *chr2 = 'y';

    // values intact and aln is correctly aligned
    assert(*chr1 == 'x' && aligned[0] == 'a' && *chr2 == 'y');
    assert((uintptr_t)aligned % 64 == 0);

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_resize(void)
{
    Arena *arena = arena_init(512, 0);

    char *str = arena_malloc(arena, 4, sizeof(*str), 1);
    assert(str);
    strcpy(str, "ab");

    // grow in place
    str = arena_resize(arena, str, 8, sizeof(*str), 1);
    assert(str && str[0] == 'a' && str[1] == 'b');
    strcpy(str + 2, "cde");

    // shrink in place
    str = arena_resize(arena, str, 4, sizeof(*str), 1);
    assert(str[0] == 'a' && str[1] == 'b' && str[2] == 'c');

    // resize to zero
    assert(arena_resize(arena, str, 0, sizeof(*str), 1) == 0);

    // last=0 behaves like arena_malloc
    char *fresh = arena_resize(arena, 0, 16, sizeof(*fresh), 1);
    assert(fresh);
    strcpy(fresh, "fresh");
    assert(fresh[0] == 'f');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_resize_grow(void)
{
    // force cross-chunk resize
    Arena *arena = arena_init(64, 1);

    char *str = arena_malloc(arena, 4, sizeof(*str), 1);
    assert(str);
    strcpy(str, "hi");

    str = arena_resize(arena, str, 80, sizeof(*str), 1);
    assert(str && str[0] == 'h' && str[1] == 'i');
    strcpy(str + 2, " world");
    assert(str[2] == ' ' && str[5] == 'r');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_resize_null(void)
{
    Arena *arena = arena_init(256, 0);

    // null pointer behaves like arena_malloc
    char *str = arena_resize(arena, 0, 16, sizeof(*str), 1);
    assert(str);
    strcpy(str, "hello");
    assert(str[0] == 'h' && str[4] == 'o');

    dump(arena, __func__);
    arena_deinit(arena);
}

static void test_resize_save_load(void)
{
    // cross-chunk resize followed by save/load frees the grown chunk
    Arena *arena = arena_init(512, 1);

    char *perm = arena_malloc(arena, 16, sizeof(*perm), 1);
    assert(perm);
    strcpy(perm, "perm");

    Mark *mark = arena_save(arena);

    // force cross-chunk resize after the mark
    char *big = arena_resize(arena, 0, 400, sizeof(*big), 1);
    assert(big);

    arena_load(arena, mark);

    // grown chunk freed; can allocate again from the original chunk
    char *fresh = arena_malloc(arena, 16, sizeof(*fresh), 1);
    assert(fresh);
    strcpy(fresh, "fresh");

    dump(arena, __func__);
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
    test_resize_null();
    test_resize_save_load();
}
