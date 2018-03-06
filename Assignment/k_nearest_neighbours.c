#include "distance.h"
#include "k_nearest_neighbours.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "sorting.h"

//make 2d struct array
dist_pt** serial_neighbours_distance(int** a, int** b, int dim, int ref_points, int query_points){

    dist_pt** point_dist = malloc(query_points*sizeof(dist_pt*));
    for(int i = 0; i<query_points; i++){
        dist_pt* row_pt_dist = malloc(ref_points*sizeof(dist_pt));
        for(int j = 0; j<ref_points; j++){
            dist_pt newDistance = {.point = j, .distance = euclidean(a[j], b[i], dim)};
            //printf("%d ",newDistance.point);
            row_pt_dist[j] = newDistance;
        }
        point_dist[i] = row_pt_dist;
    }
    //sorting segment
    double s_time = 0.0;
    printf("sorting serial\n");
    double start_time_s=omp_get_wtime();
    for(int i = 0; i<query_points; i++){
        quicksort(point_dist[i], 0, ref_points-1);
        //mergeSort(point_dist[i], 0, ref_points-1);
        s_time+=omp_get_wtime()-start_time_s;
    }
    s_time = s_time/query_points;
    printf("Serial time: %f \n", s_time);
    //mergesort(point_dist[0], 0, ref_points-1);
    return point_dist;
}

dist_pt** parallel_neighbours_distance(int** a, int** b, int dim, int ref_points, int query_points){

    dist_pt** point_dist = malloc(query_points*sizeof(dist_pt*));
    #pragma omp parallel for schedule(static, 2)
    for(int i = 0; i<query_points; i++){
        dist_pt* row_pt_dist = malloc(ref_points*sizeof(dist_pt));
        for(int j = 0; j<ref_points; j++){
            dist_pt newDistance = {.point = j, .distance = euclidean(a[j], b[i], dim)};
            //printf("%d ",newDistance.point);
            row_pt_dist[j] = newDistance;
        }
        point_dist[i] = row_pt_dist;
    }
    //sorting segment
    double p_time = 0.0;
    printf("sorting parallel\n");
    double start_time_p=omp_get_wtime();

    #pragma omp parallel for schedule(static, 8)
    for(int i = 0; i<query_points; i++){

        //quicksort_parallel_task(point_dist[i], 0, ref_points-1, ref_points);
        //mergeSortParallel_task(point_dist[i], 0, ref_points-1, ref_points);
        mergeSortParallel_sections(point_dist[i], 0, ref_points-1, ref_points);
        p_time+=omp_get_wtime()-start_time_p;
    }

    p_time = p_time/query_points;
    printf("Parallel time: %f \n", p_time);
    
    //mergesort(point_dist[0], 0, ref_points-1);
    return point_dist;
}