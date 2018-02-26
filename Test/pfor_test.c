#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
int main(int argc, char** argv){
int i;
int n = 15;
//add a schedule clause and change the setting of its value to see the effects
omp_set_num_threads(5);
//static assigns the second parameter on compilation as blocks to each thread equally, dynamic assigns the blocks to each thread on runtime its first tcome first serve.
#pragma omp parallel for schedule(static, 2)
for(i=0; i < n; i++)
{
	printf("t: %d, i: %d \n", omp_get_thread_num(), i);
}
return 0;
}

