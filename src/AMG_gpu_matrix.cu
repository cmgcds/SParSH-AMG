#include "AMG.hpp"
#include "AMG_gpu_matrix.hpp"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

using namespace std;

//Kernals
__global__ void jacobi_update(double* x,double* h,double* d,int nrow)
{
    int id = thgpu*blockIdx.x + threadIdx.x;
    if(id < nrow)
    {
        x[id] -= omega*h[id]/d[id];
    }
}


//Constructor for GPU matrix object
sp_matrix_gpu::sp_matrix_gpu(sp_matrix_mg &A)
{
    nrow = A.nrow;
    ncol = A.ncol;
    nnz = A.nnz;
    
    cudaError_t S1;
   
    S1 = cudaMalloc(&rowptr,(A.nrow+1)*sizeof(int));
    if(S1 != cudaSuccess)
    {
        cout<<"Rowpointer Allocation Failed\n";
    }
    
    S1 = cudaMalloc(&colindex,(A.nnz)*sizeof(int));
    if(S1 != cudaSuccess)
    {
        cout<<"Colindex Allocation Failed\n";
    }

    S1 = cudaMalloc(&val,(A.nnz)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"Colindex Allocation Failed\n";
    }

    S1 = cudaMalloc(&diag,(A.nrow)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"Colindex Allocation Failed\n";
    }

    cusparse_stat  = cusparseCreate(&cusparse_hndl);  
    cublas_stat    = cublasCreate(&cublas_hndl);

    cusparse_stat = cusparseCreateMatDescr(&descrA);
    cusparse_stat   = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
}


//Method to transfer matrix to GPU
void sp_matrix_gpu::matrix_transfer_gpu(sp_matrix_mg &A,cudaStream_t streams)
{
    nrow = A.nrow;
    ncol = A.ncol;
    nnz = A.nnz;

    cudaError_t S1;
    S1 = cudaMemcpyAsync(rowptr,A.rowptr,(A.nrow+1)*sizeof(int), cudaMemcpyHostToDevice,streams);
    if(S1 != cudaSuccess)
    {
        cout<<"error in 1\n";
    }

    S1 = cudaMemcpyAsync(colindex,A.colindex,(A.nnz)*sizeof(int), cudaMemcpyHostToDevice,streams);
    if(S1 != cudaSuccess)
    {
        cout<<"error in 2\n";
    }

    S1 = cudaMemcpyAsync(val,A.val,(A.nnz)*sizeof(double), cudaMemcpyHostToDevice,streams);
    if(S1 != cudaSuccess)
    {
        cout<<"error in 3\n";
    }
    if(A.nrow == A.ncol)
    {
        S1 = cudaMemcpyAsync(diag,A.diagonal,(A.nrow)*sizeof(double), cudaMemcpyHostToDevice,streams);
        if(S1 != cudaSuccess)
        {
            cout<<"error in 4\n";
        }
    }
    
}


//Method to perform Jacobi iteration on GPU
void sp_matrix_gpu::smooth_jacobi(double *bgpu,double *xgpu,double *hgpu, cudaStream_t streams,int steps)
{
    cusparse_stat  = cusparseSetStream(cusparse_hndl,streams);
    cublas_stat    = cublasSetStream(cublas_hndl,streams);

    double s1 = 1.0;
    double s2 = -1.0;
    double s3 = 0.0;
    int count = 0;
    int nblock = nrow/thgpu;

    while(count++ < steps)
    {
        cusparse_stat = cusparseDcsrmv(cusparse_hndl,transA,nrow,ncol,nnz,&s1,descrA,val,rowptr,colindex,xgpu,&s3,hgpu);
        if(cusparse_stat != CUSPARSE_STATUS_SUCCESS)
        {
            cout<<"SPMV unSuccessfull 1"<<endl;
        }
        cublas_stat = cublasDaxpy(cublas_hndl,nrow,&s2,bgpu,1,hgpu,1);
        jacobi_update<<<nblock+1,thgpu,0,streams>>>(xgpu,hgpu,diag,nrow);
    };
}

//Method for SOR smoother to be defined

 


//Destructor to free GPU memory
sp_matrix_gpu::~sp_matrix_gpu()
{
    cudaFree(rowptr);
    cudaFree(colindex);
    cudaFree(val);
    cudaFree(diag);
}

