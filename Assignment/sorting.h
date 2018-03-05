#ifndef SORTING_H
#define SORTING_H
#include "k_nearest_neighbours.h"

void quicksort(dist_pt* point_dist, int left, int right);
void mergesort(dist_pt* point_dist, int left, int right, int dim);
void insertionsort(dist_pt* point_dist, int dim);

void quicksort_parallel_task(dist_pt* point_dist, int left, int right, int ref_points);
void mergesort_parallel_task(dist_pt* point_dist, int left, int right, int dim);
void insertionsort_parallel_task(dist_pt* point_dist, int dim);


void merge2(dist_pt* arr, int l, int m, int r);
void mergeSort(dist_pt* arr, int l, int r);

void merge2Parallel(dist_pt* arr, int l, int m, int r);
void mergeSortParallel_task(dist_pt* arr, int l, int r, int ref_points);

void mergeSortParallel_sections(dist_pt* arr, int l, int r, int ref_points);

#endif