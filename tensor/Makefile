CC = gcc
LDLIBS = -lm
CFLAGS = -std=c99 -g3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion

# debug
#CFLAGS += -O0 -fno-omit-frame-pointer -fsanitize=address,undefined

# release
CFLAGS += -O3 -march=native -flto=auto -DNDEBUG -Wno-unused

# profile
#CFLAGS += -fno-omit-frame-pointer -fno-inline-functions

.PHONY: all clean test

all: main xor mnist

clean:
	@rm -f main xor mnist

test: main
	@./main

main: main.c tensor.c
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

xor: xor.c tensor.c
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

mnist: mnist.c tensor.c
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@
