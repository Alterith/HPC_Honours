#include <stdlib.h>
#include <time.h>
#include "point_set.h"

int** point_set_gen(int dim, int points){

    //create pointer array block to store all the vals
    int* elements = calloc(dim*points, sizeof(int));
    //create pointer of pointers to store each row, point number of pointers pointing to array row
    int** arrays = malloc(points*sizeof(int*));
    for(int i = 0; i<points; i++){
        //each array is of length dim
        arrays[i] = elements + i*dim;
        for(int j = 0; j < dim; j++){
            arrays[i][j] = (rand() - (RAND_MAX/2))%100;
        }
    }

    return arrays;

}