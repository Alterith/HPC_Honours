#ifndef SORTING_H
#define SORTING_H
#include "k_nearest_neighbours.h"

void quicksort(dist_pt* point_dist, int left, int right);
void mergesort(dist_pt* point_dist, int left, int right, int dim);

#endif