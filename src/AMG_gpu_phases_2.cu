#include "AMG_gpu_phases_2.hpp"
#include "AMG_gpu_phase_utilities.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <thrust/fill.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include <omp.h>

using namespace std;

void AMG_GPU1_solver::GPU_Allocations()
{
    cudaError_t S1;

    streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++)
    {
        S1 = cudaStreamCreate(&streams[i]);    
    }

    Avgpu = new sp_matrix_gpu*[l];
    Pvgpu = new sp_matrix_gpu*[l-1];
    
    Xvgpu = new double*[l];
    Bvgpu = new double*[l];
    Rvgpu = new double*[l];

    for(int i = 0;i < l;i++)
    {
        pin_AMG_MG_matrix(*Av[i],Bv[i],Xv[i],*Pv[i]);
    }

    S1 = cudaHostRegister(Xv[l],(Av[l]->nrow)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Xv[l] unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Bv[l],(Av[l]->nrow)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Bv[l] unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->rowptr,(Av[l]->nrow+1)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Row ptr unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->colindex,(Av[l]->nnz)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Col Index unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->val,(Av[l]->nnz)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Value unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->diagonal,(Av[l]->nrow)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Diagonal unsuccessfully Pinned\n";
    }

    for(int i = 0;i <= l;i++)
    {
        Avgpu[i] = new sp_matrix_gpu(*Av[i]);
        Avgpu[i]->matrix_transfer_gpu(*Av[i],streams[0]);
        if(i != 0)
        {
            cudaMalloc(&(Xvgpu[i]),(Av[i]->nrow)*sizeof(double));
            cudaMalloc(&(Bvgpu[i]),(Av[i]->nrow)*sizeof(double));
        }
        
        cudaMalloc(&(Rvgpu[i]),(Av[i]->nrow)*sizeof(double));
    }

    for(int i = 0;i < l;i++)
    {
        Pvgpu[i] = new sp_matrix_gpu(*Pv[i]);
        Pvgpu[i]->matrix_transfer_gpu(*Pv[i],streams[0]);
    }

    cudaMalloc(&(xgpu),(Av[0]->nrow)*sizeof(double));
    cudaMalloc(&(bgpu),(Av[0]->nrow)*sizeof(double));
    
}

void AMG_GPU1_solver::AMG_Solve(double* b,double* x,int iterations)
{
    Xvgpu[0] = x;
    Bvgpu[0] = b;
    int count = 0;
    double s1 = 1.0;
    double s3 = 0.0;
    double s2 = -1.0;

    double r1 = 1.0;

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
    if(iterations > 0)
    {
        while(count++ < iterations)
        {
            for(int i = 0;i < l;i++)
            {
                //SMooth 
                Avgpu[i]->smooth_jacobi(Bvgpu[i],Xvgpu[i],Rvgpu[i],streams[0],6);
                
                //Get Residual
                Avgpu[i]->cusparse_stat = cusparseDcsrmv(Avgpu[i]->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                Avgpu[i]->nrow,Avgpu[i]->ncol,Avgpu[i]->nnz,&s2,Avgpu[i]->descrA,Avgpu[i]->val,Avgpu[i]->rowptr,Avgpu[i]->colindex,Xvgpu[i],&s3,Rvgpu[i]);
                Avgpu[i]->cublas_stat   = cublasDaxpy(Avgpu[i]->cublas_hndl,Avgpu[i]->nrow,&s1,Bvgpu[i],1,Rvgpu[i],1);

                // Restrict
                Avgpu[i]->cusparse_stat = cusparseDcsrmv(Avgpu[i]->cusparse_hndl,CUSPARSE_OPERATION_TRANSPOSE,
                                Pvgpu[i]->nrow,Pvgpu[i]->ncol,Pvgpu[i]->nnz,&s1,Pvgpu[i]->descrA,Pvgpu[i]->val,Pvgpu[i]->rowptr,Pvgpu[i]->colindex,Rvgpu[i],&s3,Bvgpu[i+1]);
                thrust::fill(thrust::device,(Xvgpu[i+1]),(Xvgpu[i+1])+ Av[i+1]->nrow,0.0);
            }

            cudaError_t S1;
            S1 = cudaMemcpyAsync(Bv[l],Bvgpu[l],(Av[l]->nrow)*sizeof(double),cudaMemcpyDeviceToHost,streams[0]);
            if(S1 != cudaSuccess)
            {
                cout<<"Error in Bgpu"<<endl;
            }
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();

            Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
            S1 = cudaMemcpyAsync(Xvgpu[l],Xv[l],(Av[l]->nrow)*sizeof(double),cudaMemcpyHostToDevice,streams[0]);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();
            for(int i = l-1;i >= 0;i--)
            {
                Avgpu[i]->cusparse_stat = cusparseDcsrmv(Avgpu[i]->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                Pvgpu[i]->nrow,Pvgpu[i]->ncol,Pvgpu[i]->nnz,&s1,Pvgpu[i]->descrA,Pvgpu[i]->val,Pvgpu[i]->rowptr,Pvgpu[i]->colindex,Xvgpu[i+1],&s1,Xvgpu[i]);
                Avgpu[i]->smooth_jacobi(Bvgpu[i],Xvgpu[i],Rvgpu[i],streams[0],6);
            }

            //Get Residual
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();

            Avgpu[0]->cusparse_stat = cusparseDcsrmv(Avgpu[0]->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                Avgpu[0]->nrow,Avgpu[0]->ncol,Avgpu[0]->nnz,&s2,Avgpu[0]->descrA,Avgpu[0]->val,Avgpu[0]->rowptr,Avgpu[0]->colindex,Xvgpu[0],&s3,Rvgpu[0]);
            Avgpu[0]->cublas_stat   = cublasDaxpy(Avgpu[0]->cublas_hndl,Avgpu[0]->nrow,&s1,Bvgpu[0],1,Rvgpu[0],1);

            
            Avgpu[0]->cublas_stat   = cublasDnrm2(Avgpu[0]->cublas_hndl,Av[0]->nrow,Rvgpu[0],1,&r1);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();
        }
    }

    if(iterations == -1)
    {
        while(r1 > tol1)
        {
            for(int i = 0;i < l;i++)
            {
                //SMooth 
                Avgpu[i]->smooth_jacobi(Bvgpu[i],Xvgpu[i],Rvgpu[i],streams[0],6);
                
                //Get Residual
                Avgpu[i]->cusparse_stat = cusparseDcsrmv(Avgpu[i]->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                Avgpu[i]->nrow,Avgpu[i]->ncol,Avgpu[i]->nnz,&s2,Avgpu[i]->descrA,Avgpu[i]->val,Avgpu[i]->rowptr,Avgpu[i]->colindex,Xvgpu[i],&s3,Rvgpu[i]);
                Avgpu[i]->cublas_stat   = cublasDaxpy(Avgpu[i]->cublas_hndl,Avgpu[i]->nrow,&s1,Bvgpu[i],1,Rvgpu[i],1);

                // Restrict
                Avgpu[i]->cusparse_stat = cusparseDcsrmv(Avgpu[i]->cusparse_hndl,CUSPARSE_OPERATION_TRANSPOSE,
                                Pvgpu[i]->nrow,Pvgpu[i]->ncol,Pvgpu[i]->nnz,&s1,Pvgpu[i]->descrA,Pvgpu[i]->val,Pvgpu[i]->rowptr,Pvgpu[i]->colindex,Rvgpu[i],&s3,Bvgpu[i+1]);
                thrust::fill(thrust::device,(Xvgpu[i+1]),(Xvgpu[i+1])+ Av[i+1]->nrow,0.0);
            }

            cudaError_t S1;
            S1 = cudaMemcpyAsync(Bv[l],Bvgpu[l],(Av[l]->nrow)*sizeof(double),cudaMemcpyDeviceToHost,streams[0]);
            if(S1 != cudaSuccess)
            {
                cout<<"Error in Bgpu"<<endl;
            }

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();

            Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
            S1 = cudaMemcpyAsync(Xvgpu[l],Xv[l],(Av[l]->nrow)*sizeof(double),cudaMemcpyHostToDevice,streams[0]);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();
            for(int i = l-1;i >= 0;i--)
            {
                Avgpu[i]->cusparse_stat = cusparseDcsrmv(Avgpu[i]->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                Pvgpu[i]->nrow,Pvgpu[i]->ncol,Pvgpu[i]->nnz,&s1,Pvgpu[i]->descrA,Pvgpu[i]->val,Pvgpu[i]->rowptr,Pvgpu[i]->colindex,Xvgpu[i+1],&s1,Xvgpu[i]);
                Avgpu[i]->smooth_jacobi(Bvgpu[i],Xvgpu[i],Rvgpu[i],streams[0],6);
            }

            //Get Residual
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();

            Avgpu[0]->cusparse_stat = cusparseDcsrmv(Avgpu[0]->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                Avgpu[0]->nrow,Avgpu[0]->ncol,Avgpu[0]->nnz,&s2,Avgpu[0]->descrA,Avgpu[0]->val,Avgpu[0]->rowptr,Avgpu[0]->colindex,Xvgpu[0],&s3,Rvgpu[0]);
            Avgpu[0]->cublas_stat   = cublasDaxpy(Avgpu[0]->cublas_hndl,Avgpu[0]->nrow,&s1,Bvgpu[0],1,Rvgpu[0],1);
            
            
            
            Avgpu[0]->cublas_stat   = cublasDnrm2(Avgpu[0]->cublas_hndl,Av[0]->nrow,Rvgpu[0],1,&r1);
            
            if(print_solve_phase_details == 1)
            {
                cout<<count++<<"\t"<<r1<<endl;    
            }
            
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            cudaDeviceSynchronize();
        }
    }
    

}

void AMG_GPU1_solver::helper(double* b,double* x,int iterations)
{

    cudaMemcpy(bgpu,b,(Av[0]->nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(xgpu,x,(Av[0]->nrow)*sizeof(double),cudaMemcpyHostToDevice);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

    AMG_Solve(bgpu,xgpu,iterations);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
    
    cudaMemcpy(x,xgpu,(Av[0]->nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
}

AMG_GPU1_solver::~AMG_GPU1_solver()
{
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
     
    //Unpin Matrices
    int k = 0;
    while(k < l)
    {
        unpin_AMG_MG_matrix(*Av[k],(Bv[k]),(Xv[k]),(*Pv[k]));
        k++;  
    }

    cudaError_t S1 = cudaHostUnregister(Xv[l]);
    if(S1 != cudaSuccess)
    {
        cout<<"Xv[l] unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(Bv[l]);
    if(S1 != cudaSuccess)
    {
        cout<<"Bv[l] unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(Av[l]->rowptr);
    if(S1 != cudaSuccess)
    {
        cout<<"Row ptr unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(Av[l]->colindex);
    if(S1 != cudaSuccess)
    {
        cout<<"Col Index unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(Av[l]->val);
    if(S1 != cudaSuccess)
    {
        cout<<"Value unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(Av[l]->diagonal);
    if(S1 != cudaSuccess)
    {
        cout<<"diagonal unsuccessfully UnPinned\n";
    }

    
    for(int i = l;i > 0;i--)
    {
        Avgpu[i]->~sp_matrix_gpu();
        Pvgpu[i-1]->~sp_matrix_gpu();

        cudaFree(Xvgpu[i]);
        cudaFree(Bvgpu[i]);
        cudaFree(Rvgpu[i]);
    }

    
    Avgpu[0]->~sp_matrix_gpu();
    cudaFree(Rvgpu[0]);

    delete[] Avgpu;
    delete[] Bvgpu;
    delete[] Rvgpu;
    delete[] Xvgpu;
    delete[] Pvgpu;
    
    cudaFree(xgpu);
    cudaFree(bgpu);

}
