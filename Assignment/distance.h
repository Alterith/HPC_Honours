#ifndef DISTANCE_H
#define DISTANCE_H

double euclidean(int* a, int* b, int dim);
double manhattan(int* a, int* b, int dim);
double euclidean_parallel(int* ref, int* pt, int dim);
double manhattan_parallel(int* ref, int* pt, int dim);

#endif