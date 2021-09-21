#include <stdio.h>
#include "mpi.h"

int main(int argc, char * argv[])
{
    long N = 10000000000;
    int size, rank;
    long result=0;
    double start, stop;
    MPI_Init(&argc, &argv);
    start = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long i, j, k;
    long sum=0;
    for(i=rank+1; i<N-1; i+=size)
    {
        j = i+1; k = i+2;
        sum += (i + j + k)/2;
    }
    MPI_Reduce(&sum, &result, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    stop = MPI_Wtime();
    if(rank == 0)
    {
        printf("MPI并行结果：\nN=%lld,K=%.1lld,time=%fs\n",N,result,stop-start);
    }
    MPI_Finalize();
    return 0;
}