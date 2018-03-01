#ifndef K_NEAREST_NEIGHBOURS_H
#define K_NEAREST_NEIGHBOURS_H

//create struct to store distance to a point and the points location
struct distance_point {
    int point;
    double distance;
};
typedef struct distance_point dist_pt;

dist_pt** serial_neighbours_distance(int** a, int** b, int dim, int ref_points, int query_points);
dist_pt** parallel_neighbours_distance(int* a, int* b, int dim);

#endif