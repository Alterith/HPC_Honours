#include "distance.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

double euclidean(int* ref, int* pt, int dim){
    double distance = 0.0;
    for(int i = 0; i<dim; i++){

        //printf("t: %d, i: %d \n", omp_get_thread_num(), i);
        distance += pow(ref[i]-pt[i],2);
    }
    distance = sqrt(distance);
    return distance;
}
double manhattan(int* ref, int* pt, int dim){
    double distance = 0;
    for(int i = 0; i<dim; i++){
        distance += abs(ref[i]-pt[i]);
    }
    return distance;
}

double euclidean_parallel(int* ref, int* pt, int dim){
    double distance = 0.0;
    #pragma omp parallel for schedule(static, (dim%8)+1)
    for(int i = 0; i<dim; i++){

        //printf("t: %d, i: %d \n", omp_get_thread_num(), i);
        distance += pow(ref[i]-pt[i],2);
    }
    distance = sqrt(distance);
    return distance;
}
double manhattan_parallel(int* ref, int* pt, int dim){
    double distance = 0;
    #pragma omp parallel for schedule(static, (dim%8)+1)
    for(int i = 0; i<dim; i++){
        distance += abs(ref[i]-pt[i]);
    }
    return distance;
}