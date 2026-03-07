#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "aligned.h"

static int is_aligned(void *ptr, size_t alignment)
{
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

int main(void)
{
    for (int alignment = 1; alignment <= 1024; alignment *= 2) {
        // aligned_malloc: correct alignment
        int *ptr1 = aligned_malloc(10, sizeof(*ptr1), alignment);
        assert(ptr1 && is_aligned(ptr1, alignment));
        aligned_free(ptr1);

        // aligned_calloc: correct alignment and zero-initialized
        int *ptr2 = aligned_calloc(10, sizeof(*ptr2), alignment);
        assert(ptr2 && is_aligned(ptr2, alignment));
        for (int i = 0; i < 10; i++) {
            assert(ptr2[i] == 0);
        }

        // aligned_realloc: data preserved, new alignment applied
        for (int i = 0; i < 10; i++) {
            ptr2[i] = i;
        }
        int *ptr3 = aligned_realloc(ptr2, 20, sizeof(*ptr3), alignment);
        assert(ptr3 && is_aligned(ptr3, alignment));
        for (int i = 0; i < 10; i++) {
            assert(ptr3[i] == i);
        }
        aligned_free(ptr3);

        // aligned_realloc: null pointer behaves like aligned_malloc
        int *ptr4 = aligned_realloc(0, 10, sizeof(*ptr4), alignment);
        assert(ptr4 && is_aligned(ptr4, alignment));
        aligned_free(ptr4);

        // aligned_realloc: num=0 frees and returns 0
        int *ptr5 = aligned_malloc(10, sizeof(*ptr5), alignment);
        assert(aligned_realloc(ptr5, 0, sizeof(*ptr5), alignment) == 0);
    }
}
