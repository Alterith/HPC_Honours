#ifndef SORTING_H
#define SORTING_H
#include "k_nearest_neighbours.h"

void quicksort(dist_pt* point_dist, int left, int right);
void mergesort(dist_pt* point_dist, int left, int right, int dim);
void insertionsort(dist_pt* point_dist, int dim);

void quicksort_parallel(dist_pt* point_dist, int left, int right);
void mergesort_parallel(dist_pt* point_dist, int left, int right, int dim);
void insertionsort_parallel(dist_pt* point_dist, int dim);


#endif