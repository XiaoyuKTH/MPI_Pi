#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[]){

  int count[200];
  int local_count = 0;
  int flip;
  int rank, size; 
  double x, y, z, pi;
  double start_time, run_time;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank*100);

  flip = NUM_ITER/size;
  if(rank==0){
    flip += NUM_ITER%size;
  }

  start_time = MPI_Wtime();

  // Calculate PI following a Monte Carlo method
  for(int iter=0; iter<flip; iter++){
    // Generate random (X,Y) points
    x = (double)random()/(double)RAND_MAX;
    y = (double)random()/(double)RAND_MAX;
    z = sqrt((x*x) + (y*y));
    if(z<=1.0){
      local_count++;
    }
  }

  MPI_Gather(&local_count, 1, MPI_INT, count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(rank==0){
    for(int i=1; i<size; i++){
      local_count += count[i];
    }
    pi = ((double)local_count/(double)NUM_ITER)*4.0;
  }

  run_time = MPI_Wtime() - start_time;

  if(rank==0){
    printf("The result is %f. Time %f\n", pi, run_time);
  }

  MPI_Finalize();

  return 0;
}

