#include "defer.h"

#include <assert.h>
#include <stdlib.h>

typedef struct item {
    void *ptr;
    void (*func)(void *);
    struct item *prev;
} Item;

struct defer {
    Item *item;
};

Defer *defer_init(void)
{
    Defer *self = malloc(sizeof(*self));
    assert(self);
    self->item = 0;
    return self;
}

void defer_deinit(Defer *self)
{
    assert(self);
    Item *head = self->item;
    while (head) {
        Item *prev = head->prev;
        if (head->func) {
            head->func(head->ptr);
        }
        free(head);
        head = prev;
    }
    free(self);
}

void defer_push(Defer *self, void *ptr, void (*func)(void *))
{
    assert(self && func);
    Item *next = malloc(sizeof(*next));
    assert(next);
    next->ptr = ptr;
    next->func = func;
    next->prev = self->item;
    self->item = next;
}

void defer_pop(Defer *self, void *ptr)
{
    assert(self && ptr);
    for (Item *cur = self->item; cur; cur = cur->prev) {
        if (cur->ptr == ptr) {
            cur->func = 0;
            return;
        }
    }
}
