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
    for(int i = 0; i<query_points; i++){

        insertionsort(point_dist[i], ref_points);
    }
    
    //mergesort(point_dist[0], 0, ref_points-1);
    return point_dist;
}

dist_pt** parallel_neighbours_distance(int** a, int** b, int dim, int ref_points, int query_points){

    dist_pt** point_dist = malloc(query_points*sizeof(dist_pt*));
    #pragma omp parallel for
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
    #pragma omp parallel for schedule(static, 2)
    for(int i = 0; i<query_points; i++){

        insertionsort(point_dist[i], ref_points);
    }
    
    //mergesort(point_dist[0], 0, ref_points-1);
    return point_dist;
}