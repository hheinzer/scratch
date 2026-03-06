CC = gcc
LDLIBS =
CFLAGS = -std=c99 -g3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion

CFLAGS += -O0 -fno-omit-frame-pointer -fsanitize=address,undefined

.PHONY: all clean test

all: main

clean:
	@rm -f main

test: main
	@./main

main: main.c aligned.c aligned.h
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@
