#ifndef AMG_GPU_MATRIX_HPP_
#define AMG_GPU_MATRIX_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

//sp_matrix_gpu stores matrix on GPU
class sp_matrix_gpu
{
    public:
    
       int nrow;                        //Number of rows
       int ncol;                        //Number of columns
       int nnz;                         //Number of non-zeros
       
        //CUSPARSE handle and status
       cusparseHandle_t cusparse_hndl;
       cusparseStatus_t cusparse_stat;

       //CUBLAS handle and status
       cublasHandle_t cublas_hndl;
       cublasStatus_t cublas_stat;

        //Sparse matrix description of CUSPARSE
       cusparseMatDescr_t descrA;
       cusparseOperation_t transA;
       
       int* rowptr;                     //Row pointer array
       int* colindex;                   //Column Index array
       double *val;                     //Value array
       double *diag = nullptr;          //Diagonal array
       
    public:
        //Constructor for GPU matrix object
        sp_matrix_gpu(sp_matrix_mg &A);

        //Method to transfer matrix to GPU
        void matrix_transfer_gpu(sp_matrix_mg &A,cudaStream_t streams);

        //Method to perform Jacobi iteration on GPU
        void smooth_jacobi(double *bgpu,double *xgpu,double *hgpu, cudaStream_t streams,int steps);

        //Destructor to free GPU memory
        ~sp_matrix_gpu();
        
};


#endif