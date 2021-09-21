#include "mpi.h"
#include "omp.h"
#include <math.h>
#include <stdio.h>
#define num_steps 1000000000
#define NUM_THREADS 2
double main(int argc,char* argv[]){
    int rank, nproc;
    int i,low,up;
    double sum = 0.0, pi, step, x,t0,t1,t2;
    MPI_Status status;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nproc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    t0=MPI_Wtime();
    step = 1.0/num_steps; low = rank*(num_steps / nproc); up = low + num_steps/nproc - 1;
    #pragma omp parallel for reduction(+:sum) private(x,i)
        for (i=low;i<up; i++){ 
            x = (i-0.5)*step;
            sum += 4.0/(1.0+x*x); 
        }
    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    if(rank==0) printf("MPI+OpenMP混合结果：pi= %f ",pi*step);
    t1=MPI_Wtime();
    if(rank==0) printf("time=%f\n",t1-t0);
    MPI_Finalize();
}
