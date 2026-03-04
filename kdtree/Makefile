CC = gcc
LDLIBS = -lm
CFLAGS = -std=c99 -g3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion

# debug
CFLAGS += -O0 -fno-omit-frame-pointer -fsanitize=address,undefined

# release
#CFLAGS += -O3 -march=native -flto=auto -DNDEBUG

# profile
#CFLAGS += -fno-omit-frame-pointer -fno-inline-functions

main: main.c kdtree.c kdtree.h alloc.h
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

clean:
	@rm -rf main
