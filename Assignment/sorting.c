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
        //free(temp);
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

void quicksort_parallel_task(dist_pt* point_dist, int left, int right, int ref_points){

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
        if((right - left)<ref_points/8){
            quicksort(point_dist, left, i - 1);
            quicksort(point_dist, i + 1, right);
            
        }else{
            #pragma omp parallel num_threads(2)
		    {
			    //ensure each task only gets executed once
			    #pragma omp single
			    {
				    //task specification
				    #pragma omp task
				    quicksort_parallel_task(point_dist, left, i - 1, ref_points);
				
				    #pragma omp task
				    quicksort_parallel_task(point_dist, i + 1, right, ref_points);
			    }
		    }
        }
    	  
    }

}

void insertionsort_parallel_task(dist_pt* point_dist, int num_ref_pt){

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

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge2(dist_pt* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    dist_pt* L = malloc(n1*sizeof(dist_pt));
    dist_pt* R = malloc(n2*sizeof(dist_pt));
 
    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i].distance <= R[j].distance)
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    free(L);
    free(R);
}
 
/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(dist_pt* arr, int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l+(r-l)/2;
 
        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
 
        merge2(arr, l, m, r);
    }
}

// Merges two subarrays of arr[].
// First subarray is arr[l..m]
// Second subarray is arr[m+1..r]
void merge2Parallel(dist_pt* arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    dist_pt* L = malloc(n1*sizeof(dist_pt));
    dist_pt* R = malloc(n2*sizeof(dist_pt));


    #pragma omp parallel
		{
			//task specification
			#pragma omp for
            for (i = 0; i < n1; i++){
                L[i] = arr[l + i];
            }
				
			#pragma omp for
            for (j = 0; j < n2; j++){
                R[j] = arr[m + 1+ j];
            }
			
		}

 
    /* Copy data to temp arrays L[] and R[] */
    
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i].distance <= R[j].distance)
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}
 
/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSortParallel_task(dist_pt* arr, int l, int r, int ref_points)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l+(r-l)/2;
 
        // Sort first and second halves
        if((r-l)<(ref_points/8)){
            mergeSort(arr, l, m);
            mergeSort(arr, m+1, r);

        }else{
            #pragma omp parallel num_threads(8)
		    {
			    //ensure each task only gets executed once
			    #pragma omp single
			    {
				    //task specification
				    #pragma omp task
				    mergeSortParallel_task(arr, l, m, ref_points);
				
			    	#pragma omp task
				    mergeSortParallel_task(arr, m+1, r, ref_points);
			    }
		    }
        }
 
        merge2Parallel(arr, l, m, r);
    }
}

void mergeSortParallel_sections(dist_pt* arr, int l, int r, int ref_points)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l+(r-l)/2;
 
        // Sort first and second halves
        if((r-l)<(ref_points/8)){
            mergeSort(arr, l, m);
            mergeSort(arr, m+1, r);

        }else{
            omp_set_nested(1);
            #pragma omp parallel shared(arr)
		    {
			    //ensure each task only gets executed once
			    #pragma omp sections
			    {
				    //task specification
				    #pragma omp section
                    {
                        mergeSortParallel_sections(arr, l, m, ref_points);
                    }
			    	#pragma omp section
                    {
                        mergeSortParallel_sections(arr, m+1, r, ref_points);
                    }    
			    }
		    }
        }
 
        merge2Parallel(arr, l, m, r);
    }
}

void mergesort_parallel_task(dist_pt* point_dist, int left, int right, int dim){
    if((right-left)>0){
        double mid_1 = floor(((right+left-1)/2));
        int mid = (int)(mid_1);
        /*
        #pragma omp parallel num_threads(2)
		{
			//ensure each task only gets executed once
			#pragma omp single
			{
				//task specification
				#pragma omp task
				mergesort_parallel_task(point_dist, left, mid, dim);
				
				#pragma omp task
				mergesort_parallel_task(point_dist, mid+1, right, dim);
			}
		}*/
        mergesort_parallel_task(point_dist, left, mid, dim);
        mergesort_parallel_task(point_dist, mid+1, right, dim);
        
        
        dist_pt* temp = malloc((dim)*sizeof(dist_pt));

        #pragma omp parallel for schedule(static) ordered
        for(int i = mid; i>= left; i--){
            printf("i: %d \n", i);
            temp[i] = point_dist[i];
        }
        //#pragma omp parallel for
        for(int j = mid+1; j<=right; j++){
            //printf("right: %d left: %d (right+mid-j+1): %d \n", right, left, right+mid-j+1);
            temp[right+mid-j+1] = point_dist[j];
        }
        //printf("\n");

        int i = left;
        int j = right;
        //#pragma omp parallel for ordered
        for(int k = left; k<=right; k++){
            if(temp[i].distance<temp[j].distance){
                point_dist[k] = temp[i];
                i = i + 1;
            }else{
                point_dist[k] = temp[j];
                j = j - 1;
            }
        }
        //free(temp);
    }
}
