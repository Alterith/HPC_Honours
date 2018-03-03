#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "sorting.h"
#include "k_nearest_neighbours.h"

void quicksort(dist_pt* point_dist, int left, int right){

    if(right>left){

        dist_pt v = point_dist[right];
        int i = left;
        int j = right;

        while(i<j){
            while(point_dist[i].distance < v.distance){
                i = i + 1;
            }
            while(j>i && (point_dist[j].distance >= v.distance)){
                j = j - 1;
            }
            if(j>i){
                dist_pt t = point_dist[i];
                point_dist[i] = point_dist[j];
                point_dist[j] = t;
            }else{
                dist_pt t = point_dist[i];
                point_dist[i] = point_dist[right];
                point_dist[right] = t;
            }
        }
        quicksort(point_dist, left, i - 1);
        quicksort(point_dist, i + 1, right);

    }

}

void mergesort(dist_pt* point_dist, int left, int right, int dim){
    if((right-left)>0){
        double mid_1 = floor(((right+left-1)/2));
        int mid = (int)(mid_1);
        //printf("%d\n", (right-left));
        mergesort(point_dist, left, mid, dim);
        mergesort(point_dist, mid+1, right, dim);
        dist_pt* temp = malloc((dim)*sizeof(dist_pt));

        for(int i = mid; i>= left; i--){
            temp[i] = point_dist[i];
        }

        for(int j = mid+1; j<=right; j++){
            //printf("right: %d left: %d (right+mid-j+1): %d \n", right, left, right+mid-j+1);
            temp[right+mid-j+1] = point_dist[j];
        }
        //printf("\n");

        int i = left;
        int j = right;

        for(int k = left; k<=right; k++){
            if(temp[i].distance<temp[j].distance){
                point_dist[k] = temp[i];
                i = i + 1;
            }else{
                point_dist[k] = temp[j];
                j = j - 1;
            }
        }
        free(temp);
    }
}

void insertionsort(dist_pt* point_dist, int num_ref_pt){

    for(int i = 1; i<num_ref_pt; i++){
        dist_pt x = point_dist[i];
        int j = i - 1;

        while((j>=0) && (point_dist[j].distance>x.distance)){
            point_dist[j+1] = point_dist[j];
            j = j - 1;
        }
        point_dist[j+1] = x;
    }
}

void quicksort_parallel(dist_pt* point_dist, int left, int right){

    if(right>left){

        dist_pt v = point_dist[right];
        int i = left;
        int j = right;

        while(i<j){
            while(point_dist[i].distance < v.distance){
                i = i + 1;
            }
            while(j>i && (point_dist[j].distance >= v.distance)){
                j = j - 1;
            }
            if(j>i){
                dist_pt t = point_dist[i];
                point_dist[i] = point_dist[j];
                point_dist[j] = t;
            }else{
                dist_pt t = point_dist[i];
                point_dist[i] = point_dist[right];
                point_dist[right] = t;
            }
        }
        quicksort(point_dist, left, i - 1);
        quicksort(point_dist, i + 1, right);

    }

}

void mergesort_parallel(dist_pt* point_dist, int left, int right, int dim){
    if((right-left)>0){
        double mid_1 = floor(((right+left-1)/2));
        int mid = (int)(mid_1);
        //printf("%d\n", (right-left));
        mergesort(point_dist, left, mid, dim);
        mergesort(point_dist, mid+1, right, dim);
        dist_pt* temp = malloc((dim)*sizeof(dist_pt));

        for(int i = mid; i>= left; i--){
            temp[i] = point_dist[i];
        }

        for(int j = mid+1; j<=right; j++){
            //printf("right: %d left: %d (right+mid-j+1): %d \n", right, left, right+mid-j+1);
            temp[right+mid-j+1] = point_dist[j];
        }
        //printf("\n");

        int i = left;
        int j = right;

        for(int k = left; k<=right; k++){
            if(temp[i].distance<temp[j].distance){
                point_dist[k] = temp[i];
                i = i + 1;
            }else{
                point_dist[k] = temp[j];
                j = j - 1;
            }
        }
        free(temp);
    }
}

void insertionsort_parallel(dist_pt* point_dist, int num_ref_pt){

    for(int i = 1; i<num_ref_pt; i++){
        dist_pt x = point_dist[i];
        int j = i - 1;

        while((j>=0) && (point_dist[j].distance>x.distance)){
            point_dist[j+1] = point_dist[j];
            j = j - 1;
        }
        point_dist[j+1] = x;
    }
}