#include "defer.h"

#include <assert.h>
#include <stdlib.h>

struct defer {
    void **ptr;
    void (*func)(void *);
    Defer *prev;
};

Defer *defer_init(void)
{
    return 0;
}

void defer_deinit(Defer *self)
{
    while (self) {
        Defer *prev = self->prev;
        self->func(*self->ptr);
        free(self);
        self = prev;
    }
}

void defer_push(Defer **self, void *ptr, void (*func)(void *))
{
    assert(self && ptr && func);
    Defer *next = malloc(sizeof(*next));
    assert(next);
    next->ptr = (void **)ptr;
    next->func = func;
    next->prev = *self;
    *self = next;
}

void defer_pop(Defer **self)
{
    assert(self && *self);
    Defer *head = *self;
    *self = head->prev;
    free(head);
}
