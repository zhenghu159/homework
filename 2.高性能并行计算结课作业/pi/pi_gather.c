#include<stdio.h>
#include<mpi.h>
#include <stdlib.h>
#define N 10000000

double step=1.0/(double)N;    

int main(int argc,char *argv[]) {
    MPI_Init(&argc,&argv);
    int i, id, num_procs;
    double x,pi = 0.0;
    double buff = 0.0;
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    double sum[num_procs]; 
    for(i =id,sum[id]=0.0;i<N;i+=num_procs) {
        x =step*((double)i+0.5);
        buff += 4.0/(1.0+x*x);                                                           
    }
    MPI_Gather(&buff,1,MPI_DOUBLE,sum,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(id == 0) {
        for(i=0;i<num_procs;i++)
            pi += sum[i]*step;
        printf("MPI_Gather的结果为：pi=%f\n",pi);                          
    }
    MPI_Finalize();
    return 0;                        
}                                                               
