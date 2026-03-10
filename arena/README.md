# arena

Arena allocator written in C99. All allocations are served from a contiguous buffer; freeing is done
in bulk by resetting the arena rather than tracking individual allocations.

## Usage

Compile `arena.c` alongside your project and include `arena.h`. See `Makefile` for recommended
compilation flags.

## Functions

**`arena_init`** Create a new arena with a given capacity in bytes. Can be configured to grow
automatically when full, or to call `abort` on out-of-memory.

**`arena_deinit`** Free all memory owned by the arena, including any grown chunks.

**`arena_save`** Record the current arena position and return an opaque mark, allocated inside the
arena.

**`arena_load`** Restore the arena to a previously saved mark, reclaiming all memory allocated after
the save. Grown chunks allocated after the save are freed.

**`arena_malloc`** Allocate a number of elements of a given size and alignment. The alignment must
be a power of two; 0 uses the default. Returns null if the count is 0.

**`arena_calloc`** Like `arena_malloc`, but zero-initializes the allocation.

**`arena_resize`** Resize the most recent allocation. Grows or shrinks in place if there is room;
otherwise moves the allocation to a new chunk and copies the data. A null pointer behaves like
`arena_malloc`; a count of 0 frees the allocation and returns null.

## Implementation notes

- Alignment is implemented with a redzone-then-pad layout: each allocation is preceded by a fixed
  redzone gap followed by alignment padding, so the returned pointer is always aligned
- When built with AddressSanitizer, the redzone and any unused padding are poisoned to catch
  out-of-bounds accesses
- Growing is implemented by allocating a new chunk and swapping it into the head of a linked list,
  keeping the `Arena *` pointer stable across grows
