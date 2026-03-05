CC = gcc
LDLIBS = -lm
CFLAGS = -std=c99 -g3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion

# debug
CFLAGS += -O0 -fno-omit-frame-pointer -fsanitize=address,undefined

# release
#CFLAGS += -O3 -march=native -flto=auto -DNDEBUG -Wno-unused

# profile
#CFLAGS += -fno-omit-frame-pointer -fno-inline-functions

.PHONY: all clean

all: main libkdtree.so

clean:
	@rm -rf main libkdtree.so

main: main.c kdtree.c kdtree.h
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

libkdtree.so: kdtree.c kdtree.h
	@$(CC) $(CFLAGS) -shared -fPIC -fno-sanitize=all $^ $(LDLIBS) -o $@
