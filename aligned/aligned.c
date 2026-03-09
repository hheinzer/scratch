#include "aligned.h"

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

#else

#define MAKE_REGION_NOACCESS(addr, size) ((void)(addr), (void)(size))
#define MAKE_REGION_ADDRESSABLE(addr, size) ((void)(addr), (void)(size))

#endif

enum { ALIGN = 64 };

typedef struct {
    void *base;
    size_t bytes;
    int align;
} Header;

void *aligned_malloc(int num, int size, int align)
{
    assert(num >= 0 && size > 0 && (align & (align - 1)) == 0);

    if (num == 0) {
        return 0;
    }

    if (align == 0) {
        align = ALIGN;
    }

    size_t extra = sizeof(Header) + (align - 1);

    assert((size_t)num <= (SIZE_MAX - extra) / size);
    size_t bytes = (size_t)num * size;

    char *base = malloc(bytes + extra);
    assert(base);

    char *beg = base + sizeof(Header);
    char *ptr = beg + (-(uintptr_t)beg & (align - 1));

    ((Header *)ptr)[-1].base = base;
    ((Header *)ptr)[-1].bytes = bytes;
    ((Header *)ptr)[-1].align = align;

    MAKE_REGION_NOACCESS(base, (size_t)(ptr - base));

    return ptr;
}

void *aligned_calloc(int num, int size, int align)
{
    assert(num >= 0 && size > 0 && (align & (align - 1)) == 0);
    if (num == 0) {
        return 0;
    }
    void *ptr = aligned_malloc(num, size, align);
    return memset(ptr, 0, (size_t)num * size);
}

void *aligned_realloc(void *ptr, int num, int size, int align)
{
    assert(num >= 0 && size > 0 && (align & (align - 1)) == 0);

    if (num == 0) {
        aligned_free(ptr);
        return 0;
    }

    if (!ptr) {
        return aligned_malloc(num, size, align);
    }

    MAKE_REGION_ADDRESSABLE((char *)ptr - sizeof(Header), sizeof(Header));

    size_t old_bytes = ((Header *)ptr)[-1].bytes;
    int old_align = ((Header *)ptr)[-1].align;

    if (align == 0) {
        align = old_align;
    }

    void *new_ptr = aligned_malloc(num, size, align);

    size_t new_bytes = (size_t)num * size;
    memcpy(new_ptr, ptr, old_bytes < new_bytes ? old_bytes : new_bytes);

    free(((Header *)ptr)[-1].base);
    return new_ptr;
}

void aligned_free(void *ptr)
{
    if (ptr) {
        MAKE_REGION_ADDRESSABLE((char *)ptr - sizeof(Header), sizeof(Header));
        free(((Header *)ptr)[-1].base);
    }
}
