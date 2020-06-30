#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include "iostream"
#include "AMG_gpu_phase_utilities.hpp"

using namespace std;

//Function to pin the memory Locations for Matrix vector and rhs
void pin_AMG_MG_matrix(sp_matrix_mg& A,double* &b,double* &x,sp_matrix_mg& P)
{
    cudaError_t S1;
    S1 = cudaHostRegister(A.rowptr,(A.nrow+1)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Rowpointer unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(A.colindex,(A.nnz)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"ColumnIndex unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(A.val,(A.nnz)*sizeof(double),cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Value Array unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(A.diagonal,(A.nrow)*sizeof(double),cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Value Array unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(b,(A.nrow)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"rhs unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(x,(A.nrow)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"sol unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(P.rowptr,(P.nrow+1)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"P Rowpointer unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(P.colindex,(P.nnz)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"P ColumnIndex unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(P.val,(P.nnz)*sizeof(double),cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"P Value Array unsuccessfully Pinned\n";
    }
}

//Function to unpin the memory Locations for Matrix vector and rhs
void unpin_AMG_MG_matrix(sp_matrix_mg& A,double* &b,double* &x,sp_matrix_mg& P)
{
    cudaError_t S1;
    S1 = cudaHostUnregister(A.rowptr);
    if(S1 != cudaSuccess)
    {
        cout<<"Rowpointer unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(A.colindex);
    if(S1 != cudaSuccess)
    {
        cout<<"ColumnIndex unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(A.val);
    if(S1 != cudaSuccess)
    {
        cout<<"Value Array unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(A.diagonal);
    if(S1 != cudaSuccess)
    {
        cout<<"Value Array unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(b);
    if(S1 != cudaSuccess)
    {
        cout<<"rhs Array unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(x);
    if(S1 != cudaSuccess)
    {
        cout<<"sol Array unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(P.rowptr);
    if(S1 != cudaSuccess)
    {
        cout<<"Rowpointer unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(P.colindex);
    if(S1 != cudaSuccess)
    {
        cout<<"ColumnIndex unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(P.val);
    if(S1 != cudaSuccess)
    {
        cout<<"Value Array unsuccessfully UnPinned\n";
    }
}

//Function to swap pointers
void gpu_swap_pointers(sp_matrix_gpu* &A1,sp_matrix_gpu* &A2)
{
    sp_matrix_gpu* A3;
    A3 = A1;
    A1 = A2;
    A2 = A3;
}

//Function to compute residual on GPU
double residual(sp_matrix_gpu &A,double* b,double* x,double* h,cudaStream_t streams)
{
    A.cusparse_stat = cusparseSetStream(A.cusparse_hndl,streams);
    double s3 = 0.0;
    double s1 = 1.0;
    A.cusparse_stat = cusparseDcsrmv(A.cusparse_hndl,A.transA,A.nrow,A.ncol,A.nnz,&s1,A.descrA,A.val,A.rowptr,A.colindex,x,&s3,h);

    if(A.cusparse_stat != CUSPARSE_STATUS_SUCCESS)
    {
        cout<<"SPMV unSuccessfull"<<endl;
    }
    
    s1 = -1.0;
    A.cublas_stat = cublasSetStream(A.cublas_hndl,streams);
    A.cublas_stat = cublasDaxpy(A.cublas_hndl,A.nrow,&s1,b,1,h,1);

    if( A.cublas_stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"b - Ax unSuccessfull"<<endl;
    }

    double result;
    A.cublas_stat = cublasDnrm2(A.cublas_hndl,A.nrow,h,1,&result);
    if(A.cublas_stat != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"norm computation unsuccessfull"<<endl;
    }

    return result;
}

