//Demo of how some problems are better as serial tasks. Note that I have only implemented the parallel code and the rest has been provided by my lecturer

//To compile gcc fib.c -o fib -fopenmp
//To run ./fib number

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
int sFib(int n);
int pFib(int n);
int main(int argc, char *argv[]){
	int n, sRes, pRes, iter=10;;
	double start_time,start_time_p, s_time=0.0, p_time=0.0;
	if(argc!=2){
		printf("Enter the number n.\n");
		exit(0);
	}
	n=atoi(argv[1]);
	start_time=omp_get_wtime();
	for(int k=0; k<iter;k++){
		sRes=sFib(n);
		s_time+=omp_get_wtime()-start_time;
	}
	s_time=s_time/iter;

		/*Call the parallel function pRes here in the similar 
	way as sRes being called, and compare the performance.*/
	start_time_p=omp_get_wtime();
	for(int k=0; k<iter;k++){
		pRes=pFib(n);
		p_time+=omp_get_wtime()-start_time_p;
	}

	p_time = p_time/iter;

	/* Uncomment the following if-else statement once you 
	you complete the code */	
	if(sRes==pRes)
		printf("The %dth Fibonacci number is: %d; s: %f, p: %f, speed up: %f\n", n, pRes,s_time,p_time,s_time/p_time);
	else 
		printf("Error.\n");
	return 0;
}

int sFib(int n){
	int x, y;
	if(n<2)
		return n;
	else{
		x=sFib(n-1);
		y=sFib(n-2);
		return x+y;
	}
}

/* Parallelize the sFib() using OpenMP*/
int pFib(int n) {
    	int x, y;
		if(n<2)
			return n;
		else{
			//section to parallelize
			#pragma omp parallel num_threads(2)
			{
				//ensure each task only gets executed once
				#pragma omp single
				{
					//task specification
					#pragma omp task
					x = pFib(n-1);
					
					#pragma omp task
					y = pFib(n-2);
				}
			}

			return x+y;		
		}
	
}

//parallel code is in progress at the same time where as concurrent code is in progress at the same time but may execute at different times.