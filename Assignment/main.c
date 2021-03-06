#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "point_set.h"
#include "k_nearest_neighbours.h"


int** point_set_gen(int dim, int points);
dist_pt** serial_neighbours_distance(int** a, int** b, int dim, int ref_points, int query_points);
dist_pt** parallel_neighbours_distance(int** a, int** b, int dim, int ref_points, int query_points);


int main(void){

    int dim = 500;
    int num_ref_points = 1000000;
    int num_query_points = 10;
    double seed = time(NULL);
    
    //generate seed for random move seed to main for only 1 comparison
    srand(time(NULL));

    int** ref_set = point_set_gen(dim, num_ref_points);

    int** query_set = point_set_gen(dim, num_query_points);

    dist_pt** point_dist_serial = serial_neighbours_distance(ref_set, query_set, dim, num_ref_points, num_query_points);
    dist_pt** point_dist_parallel = parallel_neighbours_distance(ref_set, query_set, dim, num_ref_points, num_query_points);
/*
    for(int i = 0; i<num_query_points; i++){
        for(int j = 0; j<num_ref_points; j++){
            printf("Point: %d Index: %d Distance: %f \n", i, point_dist_serial[i][j].point, point_dist_serial[i][j].distance);
            printf("Point: %d Index: %d Distance: %f \n", i, point_dist_parallel[i][j].point, point_dist_parallel[i][j].distance);
        }
        printf("\n");
    }
*/
    return 0;
}