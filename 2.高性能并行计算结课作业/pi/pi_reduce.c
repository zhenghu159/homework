#include "mpi.h"
#include<stdio.h>
#include<math.h>
#define N 10000000
int main(int argc,char*argv[]){
    double sum=0.0,pi,step,x=0.0;
    int i,rank,size;
    step=1.0/N;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    for(i=rank;i<N;i+=size){
        x=(i+0.5)*step;
        sum+=4.0/(1.0+x*x);
    }
    MPI_Reduce(&sum,&pi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("MPI_Reduce结果为 pi=%f\n",pi*step);
    }
    MPI_Finalize();
    return 0;
}