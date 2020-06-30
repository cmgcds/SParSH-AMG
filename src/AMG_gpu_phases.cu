#include "AMG_gpu_phases.hpp"
#include <thrust/fill.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include "omp.h"
#include "AMG_gpu_phase_utilities.hpp"
#include "AMG.hpp"

using namespace std;
 
//Performs GPU Memory Allocations for Hybrid AMG 1 Approach
void AMG_GPU_solver::GPU_Allocations()
{
    cudaError_t S1;

    streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

    for (int i = 0; i < nstreams; i++)
    {
        S1 = cudaStreamCreate(&streams[i]);    
    }

    for(int i = 0;i < l;i++)
    {
        pin_AMG_MG_matrix(*Av[i],(Bv[i]),(Xv[i]),*Pv[i]);
    }

    A1 = new sp_matrix_gpu(*Av[0]);
    A2 = new sp_matrix_gpu(*Av[0]);

    //Pinning Extra Locations
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
        cout<<"Av[l]->rowptr unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->colindex,(Av[l]->nnz)*sizeof(int), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Av[l]->colindex unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->val,(Av[l]->nnz)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Av[l]->val unsuccessfully Pinned\n";
    }

    S1 = cudaHostRegister(Av[l]->diagonal,(Av[l]->nrow)*sizeof(double), cudaHostRegisterPortable);
    if(S1 != cudaSuccess)
    {
        cout<<"Av[l]->diagonal unsuccessfully Pinned\n";
    }


    //=====================================GPU Memory Allocations==================================/


    S1 = cudaMalloc(&xgpu,(Av[0]->nrow)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"xgpu Allocation Failed\n";
    }
    
    S1 = cudaMalloc(&bgpu,(Av[0]->nrow)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"bgpu Allocation Failed\n";
    }

    S1 = cudaMalloc(&rgpu,(Av[0]->nrow)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"rgpu Allocation Failed\n";
    }

    S1 = cudaMalloc(&hgpu,(Av[0]->nrow)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"hgpu Allocation Failed\n";
    }

    S1 = cudaMalloc(&P1valgpu,(Av[0]->nrow)*sizeof(double));
    if(S1 != cudaSuccess)
    {
        cout<<"P1valgpu Allocation Failed\n";
    }

    S1 = cudaMalloc(&P1rowgpu,(Av[0]->nrow+1)*sizeof(int));
    if(S1 != cudaSuccess)
    {
        cout<<"P1rowgpu Allocation Failed\n";
    }

    S1 = cudaMalloc(&P1colgpu,(Av[0]->nrow)*sizeof(int));
    if(S1 != cudaSuccess)
    {
        cout<<"P1colgpu Allocation Failed\n";
    }

    A2->matrix_transfer_gpu(*Av[0],streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
}
 
//Solve Phase of AMG rhs and solution vector on CPU
void AMG_GPU_solver::AMG_GPU_solve(double* b,double* x,int iterations)
{
    cudaError_t S1;

    std::copy(b,b+Av[0]->nrow,Bv[0]);
    std::copy(x,x+Av[0]->nrow,Xv[0]);
    
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    S1 = cudaMemcpyAsync(xgpu,Xv[0],(Av[0]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
    if(S1 != cudaSuccess)
    {
        cout<<"error in xgpu\n";
    }

    S1 = cudaMemcpyAsync(bgpu,Bv[0],(Av[0]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
    if(S1 != cudaSuccess)
    {
        cout<<"error in bgpu\n";
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

     
    double res = 1.0;
    res = residual(*A1,bgpu,xgpu,hgpu,streams[0]); 

    int cycles = 0;
    
    if (iterations == -1)
    {
        while(res > tol1)
        {
            int k1 = 0;
            while(k1 < l)
            {
                gpu_swap_pointers(A1,A2);
                smooth_and_get_prolongator(k1);
                restrict_and_get_next(k1);
                k1++;
            }

            Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
               
            S1 = cudaMemcpyAsync(xgpu,Xv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
            S1 = cudaMemcpyAsync(bgpu,Bv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[1]);
    
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
    
            S1 = cudaMemcpyAsync(rgpu,Xv[l],    (Av[l]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
           
            double s3 = 1.0; 
    
            A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,Pv[l-1]->nrow,Pv[l-1]->ncol,Pv[l-1]->nnz,&s3,A1->descrA,P1valgpu,P1rowgpu,P1colgpu,rgpu,&s3,xgpu);
            if(A1->cusparse_stat != CUSPARSE_STATUS_SUCCESS)
            {
                cout<<"SPMV unSuccessfull"<<endl;
            }
    
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            
            k1 = l - 2;
            while(k1 >= 0)
            {
                smooth_and_prolongate(k1);
                gpu_swap_pointers(A1,A2);
                cudaStreamSynchronize(streams[0]);
                cudaStreamSynchronize(streams[1]);
                k1 = k1 - 1;
            }
    
            A1->smooth_jacobi(bgpu,xgpu,hgpu,streams[0],smooth_iter);
    
            res = residual(*A1,bgpu,xgpu,hgpu,streams[0]); 

            if(print_solve_phase_details == 1)
            {
                cout<<cycles<<" "<<res<<endl;
            }	
            cycles++;
            gpu_swap_pointers(A1,A2);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
        }
 
        S1 = cudaMemcpyAsync(Xv[0],xgpu,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToHost,streams[0]);
        if(S1 != cudaSuccess)
        {
            cout<<"error in xgpu\n";
        }

        std::copy(Xv[0],Xv[0]+Av[0]->nrow,x);
    }
    else
    {
        while(cycles < iterations)
        {
            int k1 = 0;
            while(k1 < l)
            {
                gpu_swap_pointers(A1,A2);
                smooth_and_get_prolongator(k1);
                restrict_and_get_next(k1);
                k1++;
            }
    
            Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
               
            S1 = cudaMemcpyAsync(xgpu,Xv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
            S1 = cudaMemcpyAsync(bgpu,Bv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[1]);
    
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
    
            S1 = cudaMemcpyAsync(rgpu,Xv[l],    (Av[l]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
           
            double s3 = 1.0; 
    
            A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,Pv[l-1]->nrow,Pv[l-1]->ncol,Pv[l-1]->nnz,&s3,A1->descrA,P1valgpu,P1rowgpu,P1colgpu,rgpu,&s3,xgpu);
            if(A1->cusparse_stat != CUSPARSE_STATUS_SUCCESS)
            {
                cout<<"SPMV unSuccessfull"<<endl;
            }
    
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            
            k1 = l - 2;
            
            while(k1 >= 0)
            {
                smooth_and_prolongate(k1);
                gpu_swap_pointers(A1,A2);
                cudaStreamSynchronize(streams[0]);
                cudaStreamSynchronize(streams[1]);
                k1 = k1 - 1;
            }
    
            A1->smooth_jacobi(bgpu,xgpu,hgpu,streams[0],smooth_iter); 
            cycles++; 
            gpu_swap_pointers(A1,A2);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
        }

        S1 = cudaMemcpyAsync(Xv[0],xgpu,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToHost,streams[0]);
        if(S1 != cudaSuccess)
        {
            cout<<"error in xgpu\n";
        }
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        std::copy(Xv[0],Xv[0]+Av[0]->nrow,x);
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
}

//Solve Phase of AMG rhs and solution vector on GPU
void AMG_GPU_solver::AMG_GPU_solve_1(double* b,double* x,int iterations)
{
    cudaError_t S1;
    
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    S1 = cudaMemcpyAsync(Xv[0],x,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToHost,  streams[0]);
    S1 = cudaMemcpyAsync(xgpu, x,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToDevice,streams[1]);

    if(S1 != cudaSuccess)
    {
        cout<<"error in xgpu\n";
    }

    S1 = cudaMemcpyAsync(Bv[0],b,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToHost,  streams[0]);
    S1 = cudaMemcpyAsync(bgpu, b,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToDevice,streams[1]);

    if(S1 != cudaSuccess)
    {
        cout<<"error in xgpu\n";
    }
    

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

     
    double res = 1.0;
    res = residual(*A2,bgpu,xgpu,hgpu,streams[0]); 

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

    int cycles = 0;
    if (iterations == -1)
    {
        while(res > tol1)
        {
             int k1 = 0;
             while(k1 < l)
             {
                 gpu_swap_pointers(A1,A2);
                 smooth_and_get_prolongator(k1);
                 restrict_and_get_next(k1);
                 k1++;
             }
  
             Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
              
             S1 = cudaMemcpyAsync(xgpu,Xv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
             S1 = cudaMemcpyAsync(bgpu,Bv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[1]);
    
             cudaStreamSynchronize(streams[0]);
             cudaStreamSynchronize(streams[1]);
    
             S1 = cudaMemcpyAsync(rgpu,Xv[l],    (Av[l]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
           
             double s3 = 1.0; 
    
             A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,Pv[l-1]->nrow,Pv[l-1]->ncol,Pv[l-1]->nnz,&s3,A1->descrA,P1valgpu,P1rowgpu,P1colgpu,rgpu,&s3,xgpu);
             if(A1->cusparse_stat != CUSPARSE_STATUS_SUCCESS)
             {
                 cout<<"SPMV unSuccessfull"<<endl;
             }
    
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            
            k1 = l - 2;
            while(k1 >= 0)
            {
                smooth_and_prolongate(k1);
                gpu_swap_pointers(A1,A2);
                cudaStreamSynchronize(streams[0]);
                cudaStreamSynchronize(streams[1]);
                k1 = k1 - 1;
            }
    
            A1->smooth_jacobi(bgpu,xgpu,hgpu,streams[0],6);
    
            res = residual(*A1,bgpu,xgpu,hgpu,streams[0]); 

            if(print_solve_phase_details == 1)
            {
                cout<<cycles<<" "<<res<<endl;
            }	

            cycles++;
            gpu_swap_pointers(A1,A2);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
         }

        S1 = cudaMemcpyAsync(Xv[0],xgpu,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToHost,streams[0]);
        if(S1 != cudaSuccess)
        {
            cout<<"error in xgpu\n";
        }
        
        S1 = cudaMemcpyAsync(x, xgpu,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToDevice,streams[1]);
    }
    else
    {
         while(cycles < iterations)
         {
            int k1 = 0;
             while(k1 < l)
             {
                 gpu_swap_pointers(A1,A2);
                 smooth_and_get_prolongator(k1);
                 restrict_and_get_next(k1);
                 k1++;
             }
    
             Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
               
             S1 = cudaMemcpyAsync(xgpu,Xv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
             S1 = cudaMemcpyAsync(bgpu,Bv[l-1],(Av[l-1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[1]);
    
             cudaStreamSynchronize(streams[0]);
             cudaStreamSynchronize(streams[1]);
    
             S1 = cudaMemcpyAsync(rgpu,Xv[l],    (Av[l]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[0]);
           
             double s3 = 1.0; 
    
             A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,CUSPARSE_OPERATION_NON_TRANSPOSE,Pv[l-1]->nrow,Pv[l-1]->ncol,Pv[l-1]->nnz,&s3,A1->descrA,P1valgpu,P1rowgpu,P1colgpu,rgpu,&s3,xgpu);
             if(A1->cusparse_stat != CUSPARSE_STATUS_SUCCESS)
             {
                 cout<<"SPMV unSuccessfull"<<endl;
             }
    
            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
            
            k1 = l - 2;
            
            while(k1 >= 0)
            {
                smooth_and_prolongate(k1);
                gpu_swap_pointers(A1,A2);
                cudaStreamSynchronize(streams[0]);
                cudaStreamSynchronize(streams[1]);
                k1 = k1 - 1;
            }
    
            A1->smooth_jacobi(bgpu,xgpu,hgpu,streams[0],6);
    
            res = residual(*A1,bgpu,xgpu,hgpu,streams[0]); 
            cycles++;
            
            gpu_swap_pointers(A1,A2);

            cudaStreamSynchronize(streams[0]);
            cudaStreamSynchronize(streams[1]);
         }

        S1 = cudaMemcpyAsync(Xv[0],xgpu,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToHost,streams[0]);
        if(S1 != cudaSuccess)
        {
            cout<<"error in xgpu\n";
        }
        S1 = cudaMemcpyAsync(x, xgpu,(Av[0]->nrow)*sizeof(double), cudaMemcpyDeviceToDevice,streams[1]);
    }
    cudaDeviceSynchronize();
}

//Helper functions for hybrid AMG 1 Approach


//Method to overlap transfer of prolongator and smoothing iterations while going down the V cycle
void AMG_GPU_solver::smooth_and_get_prolongator(int k1)
{
    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat   = cublasSetStream(A1->cublas_hndl,streams[0]);

    double s1 = 1.0;
    double s2 = -1.0;
    double s3 = 0.0;

    A2->nrow = Av[k1+1]->nrow;
    A2->ncol = Av[k1+1]->ncol;
    A2->nnz  = Av[k1+1]->nnz;

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);

    cudaMemcpyAsync(P1colgpu,    Pv[k1]->colindex, (Pv[k1]->nnz)*sizeof(int),    cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(P1rowgpu,    Pv[k1]->rowptr,   (Pv[k1]->nrow+1)*sizeof(int),   cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(P1valgpu,    Pv[k1]->val,      (Pv[k1]->nnz)*sizeof(double),    cudaMemcpyHostToDevice,streams[1]);
    
    A2->matrix_transfer_gpu(*Av[k1+1],streams[2]);
    A1->smooth_jacobi(bgpu,xgpu,hgpu,streams[0],smooth_iter);
   
    A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,A1->descrA,A1->val,A1->rowptr,A1->colindex,xgpu,&s3,hgpu);
    A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A1->nrow,&s2,bgpu,1,hgpu,1);
    
    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,NULL);
    A1->cublas_stat   = cublasSetStream  (A1->cublas_hndl,NULL);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
}

//Method to overlap restriction operations and getting next matrix on GPU
void AMG_GPU_solver::restrict_and_get_next(int k1)
{
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
   
    cudaMemcpyAsync(Xv[k1],xgpu, (A1->nrow)*sizeof(double), cudaMemcpyDeviceToHost,streams[0]);
    thrust::fill_n(thrust::cuda::par.on(streams[0]),xgpu,Av[0]->nrow,0.0);
    
    double s1 = -1.0;
    double s2 = 0.0;
   
    A1->cusparse_stat   = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cusparse_stat   = cusparseDcsrmv(A1->cusparse_hndl,CUSPARSE_OPERATION_TRANSPOSE,Pv[k1]->nrow,Pv[k1]->ncol,Pv[k1]->nnz,&s1,A1->descrA,P1valgpu,P1rowgpu,P1colgpu,hgpu,&s2,bgpu);
    if(A1->cusparse_stat != CUSPARSE_STATUS_SUCCESS)
    {
        cout<<"SPMV unSuccessfull"<<endl;
    }

    cudaMemcpyAsync(Bv[k1+1],bgpu, (Pv[k1]->ncol)*sizeof(double), cudaMemcpyDeviceToHost,streams[0]);
    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,NULL);
    
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
}

//Method to overlap smoothing iteration and transfer of prolongation matrix while going up the V cycle
void AMG_GPU_solver::smooth_and_prolongate(int k1)
{
    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat   = cublasSetStream  (A1->cublas_hndl,streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  
    cudaMemcpyAsync(P1colgpu,    Pv[k1]->colindex, (Pv[k1]->nnz)*sizeof(int),    cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(P1rowgpu,    Pv[k1]->rowptr,   (Pv[k1]->nrow+1)*sizeof(int), cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(P1valgpu,    Pv[k1]->val,      (Pv[k1]->nnz)*sizeof(double), cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(rgpu,        Xv[k1],           (Pv[k1]->nrow)*sizeof(double),cudaMemcpyHostToDevice,streams[1]);
   
    A2->matrix_transfer_gpu(*Av[k1],streams[2]);
   
    A1->smooth_jacobi(bgpu,xgpu,hgpu,streams[0],smooth_iter);
   
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    double s1 = 1.0;
  
    A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,Pv[k1]->nrow,Pv[k1]->ncol,Pv[k1]->nnz,&s1,A1->descrA,P1valgpu,P1rowgpu,P1colgpu,xgpu,&s1,rgpu);
    if(A1->cusparse_stat != CUSPARSE_STATUS_SUCCESS)
    {
        cout<<"SPMV unSuccessfull 1"<<endl;
    }

    cudaMemcpyAsync(xgpu,rgpu, (Pv[k1]->nrow)*sizeof(double), cudaMemcpyDeviceToDevice,streams[0]);
    cudaMemcpyAsync(bgpu,Bv[k1],(Pv[k1]->nrow)*sizeof(double), cudaMemcpyHostToDevice,streams[1]);

    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,NULL);
    A1->cublas_stat   = cublasSetStream  (A1->cublas_hndl,NULL);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
}


AMG_GPU_solver::~AMG_GPU_solver()
{
    
    //Releasing Resources
    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    
    int k = 0;
    while(k < l)
    {
        unpin_AMG_MG_matrix(*Av[k],(Bv[k]),(Xv[k]),(*Pv[k]));
        k++;
    }

    
    A1->~sp_matrix_gpu();
    A2->~sp_matrix_gpu();
    
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
        cout<<"Av[l]->rowptr unsuccessfully UnPinned\n";
    }
 
    S1 = cudaHostUnregister(Av[l]->colindex);
    if(S1 != cudaSuccess)
    {
        cout<<"Av[l]->colindex unsuccessfully UnPinned\n";
    }

    S1 = cudaHostUnregister(Av[l]->val);
    if(S1 != cudaSuccess)
    {
        cout<<"Av[l]->val unsuccessfully UnPinned\n";
    }
    
    S1 = cudaHostUnregister(Av[l]->diagonal);
    if(S1 != cudaSuccess)
    {
        cout<<"Av[l]->diagonal unsuccessfully UnPinned\n";
    }

    cudaFree(xgpu);
    cudaFree(bgpu);
    cudaFree(rgpu);
    cudaFree(hgpu);
    cudaFree(P1valgpu);
    cudaFree(P1rowgpu);
    cudaFree(P1colgpu);
    
}







