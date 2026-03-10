#include <assert.h>
#include <stdlib.h>

#include "defer.h"

static void test_basic(void)
{
    Defer *defer = defer_init();

    int *first = malloc(sizeof(*first));
    assert(first);
    defer_push(&defer, &first, free);

    int *second = malloc(sizeof(*second));
    assert(second);
    defer_push(&defer, &second, free);

    // free manually (set to zero to avoid double free)
    free(second);
    second = 0;

    defer_deinit(defer);
}

static void test_pop(void)
{
    Defer *defer = defer_init();

    int *ptr = malloc(sizeof(*ptr));
    assert(ptr);
    defer_push(&defer, &ptr, free);

    // cancel defer, free manually
    defer_pop(&defer);
    free(ptr);

    defer_deinit(defer);
}

static void test_deinit_null(void)
{
    Defer *defer = defer_init();

    assert(!defer);

    defer_deinit(defer);
}

int main(void)
{
    test_basic();
    test_pop();
    test_deinit_null();
}
