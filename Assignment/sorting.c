#include <stdio.h>
#include <stdlib.h>
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