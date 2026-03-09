CC = gcc
LDLIBS =
CFLAGS = -std=c99 -g3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion

CFLAGS += -O0 -fno-omit-frame-pointer -fsanitize=address,undefined

.PHONY: all clean test

all: main

clean:
	@rm -f main

test: main
	@./main

main: main.c arena.c arena.h
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@
