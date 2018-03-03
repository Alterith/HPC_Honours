CC=gcc
CFLAGS= -Wall -fopenmp -g -O0
LIBS = -lm

all: clean main

main: main.c point_set.o point_set.h distance.o distance.h k_nearest_neighbours.o k_nearest_neighbours.h sorting.o sorting.h
	cc -o main point_set.o distance.o k_nearest_neighbours.o sorting.o main.c -lm
clean:
	rm -f *.out *.o

