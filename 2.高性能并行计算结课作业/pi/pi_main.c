#include<stdio.h>
#include"pi_function.h"

void main(){
double speedup,efficiency;
printf("请选择一种方法：\n[1]parallel [2]padding [3]firstprivate [4]barrier [5]critical [6]atomic [7]for\n");
int methods;
scanf("%d",&methods);

//串行时间
double T0=0;
T0=serial();

int N[8]={2,4,6,8,10,20,30,40};
for (int a=0;a<8;a++){

    //1.#pragma omp parallel 
    if(methods==1){
        double T1=parallel(N[a]);
        speedup=T0/T1;
        efficiency=speedup/N[a]*100;
    }

    //2.加padding的#pragma omp parallel 
    else if(methods==2){
        double T2=padding(N[a]);
        speedup=T0/T2;
        efficiency=speedup/N[a]*100;
    }

    //3.#pragma omp parallel firstprivate
    else if(methods==3){
        double T3=firstprivate(N[a]);
        speedup=T0/T3;
        efficiency=speedup/N[a]*100;
    }
    
    //4.#pragma omp barrier
    else if(methods==4){
        double T4=barrier(N[a]);
        speedup=T0/T4;
        efficiency=speedup/N[a]*100;
    }

    //5.#pragma omp critical
    else if(methods==5){
        double T5=critical(N[a]);
        speedup=T0/T5;
        efficiency=speedup/N[a]*100;
    }

    //6.#pragma omp atomic
    else if(methods==6){
        double T6=atomic(N[a]);
        speedup=T0/T6;
        efficiency=speedup/N[a]*100;
    }

    //7.#pragma omp parallel for
    else if(methods==7){
        double T7=parallelfor(N[a]);
        speedup=T0/T7;
        efficiency=speedup/N[a]*100;
    }

    //输入错误，请重新输入
    else{
        printf("输入错误，请重新输入");
    }
    
    printf("speedup=%.2f,efficiency=%.2f%%\n",speedup,efficiency);
}
}