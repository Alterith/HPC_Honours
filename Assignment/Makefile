CC=gcc
CFLAGS= -Wall -fopenmp
LIBS = -lm

all: clean main

main: main.c point_set.o point_set.h distance.o distance.h k_nearest_neighbours.o k_nearest_neighbours.h
	cc -o main point_set.o distance.o k_nearest_neighbours.o main.c -lm
clean:
	rm -f *.out *.o

