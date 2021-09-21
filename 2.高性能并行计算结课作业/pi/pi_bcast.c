#include<stdio.h>
#include<mpi.h>
double main(int argc, char *argv[]){
    int id,n,i,N=0;
    double sum,step,x,pi,mypi;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    if(id==0){
        scanf("%d",&N); 
    }
    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    step=1.0/N;
    sum=0.0;
    for(int i=id;i<N; i=i+n){
        x =(i-0.5)*step;
        sum+=4.0/(1.0+x*x);
    }
    pi+=sum*step;
    MPI_Reduce(&pi,&mypi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(id==0){
    printf("MPI_Bcast结果为 pi=%f\n",mypi);
    }
    MPI_Finalize();
    return 0;
}

