#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static inline void *alloc(int num, size_t size)
{
    assert(num >= 0 && size > 0);
    if (num == 0) {
        return 0;
    }
    assert((size_t)num <= SIZE_MAX / size);
    void *ptr = malloc((size_t)num * size);
    assert(ptr);
    return ptr;
}
