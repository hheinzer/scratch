#include <assert.h>
#include <stdlib.h>

#include "defer.h"

static void test_basic(void)
{
    // push two items and let deinit free both
    Defer *defer = defer_init();

    int *first = malloc(sizeof(*first));
    assert(first);
    defer_push(defer, first, free);

    int *second = malloc(sizeof(*second));
    assert(second);
    defer_push(defer, second, free);

    defer_deinit(defer);
}

static void test_pop(void)
{
    // pop cancels the deferred call; the resource must be freed manually
    Defer *defer = defer_init();

    int *ptr = malloc(sizeof(*ptr));
    assert(ptr);
    defer_push(defer, ptr, free);

    defer_pop(defer, ptr);
    free(ptr);

    defer_deinit(defer);
}

static void test_pop_middle(void)
{
    // pop can cancel any item in the stack, not just the top
    Defer *defer = defer_init();

    int *first = malloc(sizeof(*first));
    assert(first);
    defer_push(defer, first, free);

    int *second = malloc(sizeof(*second));
    assert(second);
    defer_push(defer, second, free);

    int *third = malloc(sizeof(*third));
    assert(third);
    defer_push(defer, third, free);

    defer_pop(defer, second);
    free(second);

    defer_deinit(defer);
}

static void test_pop_not_found(void)
{
    // pop is a no-op if the pointer is not in the stack
    Defer *defer = defer_init();

    int *ptr = malloc(sizeof(*ptr));
    assert(ptr);
    defer_push(defer, ptr, free);

    int *other = malloc(sizeof(*other));
    defer_pop(defer, other);
    free(other);

    defer_deinit(defer);
}

static int *alloc_int(Defer *defer)
{
    int *ptr = malloc(sizeof(*ptr));
    assert(ptr);
    defer_push(defer, ptr, free);
    return ptr;
}

static void test_returned_from_function(void)
{
    // a subroutine can push onto a caller-owned defer stack
    Defer *defer = defer_init();

    int *ptr = alloc_int(defer);
    assert(ptr);

    defer_deinit(defer);
}

int main(void)
{
    test_basic();
    test_pop();
    test_pop_middle();
    test_pop_not_found();
    test_returned_from_function();
}
