# aligned

Aligned memory allocation in C99. Provides `malloc`, `calloc`, `realloc`, and `free` variants that
guarantee a caller-specified alignment.

## Usage

Compile `aligned.c` alongside your project and include `aligned.h`. See `Makefile` for recommended
compilation flags.

## Functions

**`aligned_malloc`** Allocate a number of elements with a given alignment (must be a power of 2; 0
uses the default of 64). Returns null if the count is 0. Free with `aligned_free`.

**`aligned_calloc`** Like `aligned_malloc`, but zero-initializes the allocation.

**`aligned_realloc`** Resize a previous allocation. Data is preserved up to the smaller of the old
and new sizes. If the alignment is 0, the original alignment is preserved. A null pointer behaves
like `aligned_malloc`; a count of 0 frees the pointer and returns null.

**`aligned_free`** Free a pointer returned by any of the above functions.

## Implementation notes

- A header storing the original `malloc` pointer, the allocation size, and the alignment is placed
  immediately before the returned pointer
- When built with AddressSanitizer, the padding between the header and the aligned pointer is
  poisoned to catch out-of-bounds accesses into it
