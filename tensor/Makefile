CC = gcc
LDLIBS = -lm
CFLAGS = -std=c99 -g3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion

# debug
CFLAGS += -O0 -fno-omit-frame-pointer -fsanitize=address,undefined

# release
#CFLAGS += -O3 -march=native -flto=auto -DNDEBUG -Wno-unused

# profile
#CFLAGS += -fno-omit-frame-pointer -fno-inline-functions

.PHONY: all clean test

all: main

clean:
	@rm -f main

test: main
	@./main

main: main.c tensor.c tensor.h
	@$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@
