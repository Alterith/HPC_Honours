CC=gcc
CFLAGS= -Wall -fopenmp -g -O0
LIBS = -lm

all: clean main

valgrind: valgrind_clean
	valgrind --leak-check=full \
	--show-leak-kinds=all \
	--track-origins=yes \
	--verbose \
	--log-file=valgrind-out.txt \
	./main.o

valgrind_clean:
	rm -f *.txt

main: main.c point_set.o point_set.h distance.o distance.h k_nearest_neighbours.o k_nearest_neighbours.h sorting.o sorting.h
	cc -o main.o point_set.o distance.o k_nearest_neighbours.o sorting.o main.c -lm -fopenmp

clean:
	rm -f *.out *.o *.txt

