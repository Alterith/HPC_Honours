#include "distance.h"
#include <math.h>
#include <stdlib.h>

double euclidean(int* ref, int* pt, int dim){
    double distance = 0.0;
    for(int i = 0; i<dim; i++){
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
