#include "AMG.hpp"
#include "AMG_phases.hpp"
#include "AMG_gpu_phases.hpp"
#include "AMG_coarsening.hpp"
#include "AMG_gpu_phases_2.hpp"
#include "AMG_cycle_utilities.hpp"
#include "iostream"
#include "omp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

using namespace std;
//Kernals for daxpby and daxpbypc

__global__ void daxpby(int n, double alpha, double *x, double beta, double* y)
{
    int id = thgpu*blockIdx.x + threadIdx.x;
    if(id < n)
    {
        y[id] = alpha*x[id]+beta*y[id];
    }
}

__global__ void daxpbyc(int n, double alpha, double *x, double beta, double* y,double gamma,double* c)
{
    int id = thgpu*blockIdx.x + threadIdx.x;
    if(id < n)
    {
        c[id] = alpha*x[id]+beta*y[id]+gamma*c[id];
    }
}

void Solver_CG_2(sp_matrix_mg& A,double* &b,double* &x)
{
    cudaError_t S1;
    cudaStream_t *streams;

    streams = (cudaStream_t *) malloc(2 * sizeof(cudaStream_t));

    for (int i = 0; i < 2; i++)
    {
        S1 = cudaStreamCreate(&streams[i]);
         if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Creation"<<endl;
        }
    }

    sp_matrix_gpu *A1 = new sp_matrix_gpu(A);
    A1->matrix_transfer_gpu(A,streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    double *Apgpu;
    double *pgpu;

    double *xgpu;
    double *bgpu;
    double *rgpu;

    S1 = cudaMalloc(&Apgpu,(A.nrow)*sizeof(double));
    S1 = cudaMalloc(&pgpu,(A.nrow)*sizeof(double));
    S1 = cudaMalloc(&xgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&bgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&rgpu, (A.nrow)*sizeof(double));

    cudaMemcpy(xgpu,x,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(bgpu,b,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);

    double beta;
    double alpha;
    double alpha1;

    int count = 0;

    double s1 = 1.0;
    double s3 = 0.0;

    A1->cusparse_stat  = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat    = cublasSetStream  (A1->cublas_hndl,  streams[0]);
    A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,xgpu,&s3,rgpu);

    daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,bgpu,-1,rgpu);


    //cudaMemcpy(rgpu,bgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(pgpu,bgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);

    double r1;
    cublasDnrm2(A1->cublas_hndl,A.nrow,rgpu,1,&r1);

    //cout<<r1<<endl;
    count = 0;
    while(r1 > tol1)
    {
        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,pgpu,&s3,Apgpu);

        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,pgpu,1,Apgpu,1,&alpha);
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,rgpu,1,rgpu, 1,&alpha1);

        alpha = alpha1/alpha;

        A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A.nrow,&alpha,pgpu,1,xgpu,1);
        alpha = -alpha;
        A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A.nrow,&alpha,Apgpu,1,rgpu,1);

        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,rgpu,1,rgpu,1 ,&beta);
        beta = beta/alpha1;

        daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,rgpu,beta,pgpu);

        r1 = sqrt(beta*alpha1);
        
        if(print_solve_phase_details == 1)
        {
            cout<<count++<<"\t"<<r1<<endl;
        }
        
    }

    cudaMemcpy(x,xgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(Apgpu);
    cudaFree(pgpu);
    cudaFree(xgpu);
    cudaFree(bgpu);
    cudaFree(rgpu);

    A1->~sp_matrix_gpu();

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

//AMG-PCG 3 Implementation
void Solver_PCG_3(sp_matrix_mg& A,double* &b,double* &x)
{
    AMG_GPU_solver* G_Solver = new AMG_GPU_solver();
	G_Solver->AMG_solver_setup_jacobi(A);
	G_Solver->GPU_Allocations();
    
    double* z0       = new double[A.nrow]();
	double*	r0       = new double[A.nrow];

    cudaError_t S1;
    cudaStream_t *streams;

    streams = (cudaStream_t *) malloc(2 * sizeof(cudaStream_t));

    for (int i = 0; i < 2; i++)
    {
        S1 = cudaStreamCreate(&streams[i]); 
        if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Creation"<<endl;
        }   
    }

    sp_matrix_gpu *A1 = new sp_matrix_gpu(A);
    A1->matrix_transfer_gpu(A,streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

    double *Apgpu;
    double *pgpu;

    double *xgpu;
    double *bgpu;
    double *rgpu;

    double *zgpu;

    S1 = cudaMalloc(&Apgpu,(A.nrow)*sizeof(double));
    S1 = cudaMalloc(&pgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&xgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&bgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&rgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&zgpu, (A.nrow)*sizeof(double));

    cudaMemcpy(xgpu,x,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(bgpu,b,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(r0,bgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    
    double beta;
    double alpha;
    double alpha1;

    int count = 0;

    double s1 = 1.0;
    double s3 = 0.0;

    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat   = cublasSetStream  (A1->cublas_hndl,  streams[0]);

    cudaMemcpy(rgpu,bgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(pgpu,bgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);

    double r1;
    cublasDnrm2(A1->cublas_hndl,A.nrow,rgpu,1,&r1);
    
   
    thrust::fill_n(thrust::device,zgpu,A.nrow,0.0);
    G_Solver->AMG_GPU_solve_1(rgpu,zgpu,1);
    cudaMemcpy(pgpu,zgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);

    count = 0;
    cudaDeviceSynchronize();

    while(r1 > tol1)
    {
        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                               A1->descrA,A1->val,A1->rowptr,A1->colindex,pgpu,&s3,Apgpu);
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,pgpu,1,Apgpu,1,&alpha);
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,rgpu,1,zgpu, 1,&alpha1);
        alpha = alpha1/alpha;
        A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A.nrow,&alpha,pgpu,1,xgpu,1);
        alpha = -alpha;
        A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A.nrow,&alpha,Apgpu,1,rgpu,1);
        thrust::fill_n(thrust::device,zgpu,A.nrow,0.0);
        G_Solver->AMG_GPU_solve_1(rgpu,zgpu,1);
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,rgpu,1,zgpu,1 ,&beta);
        beta = beta/alpha1;
        daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,zgpu,beta,pgpu);
        r1 = sqrt(beta*alpha1);
        count++;

        if(print_solve_phase_details == 1)
        {
            std::cout<<count<<"\t"<<r1<<"\n";
        }	

    }
    
    cudaMemcpy(x,xgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(zgpu);
    cudaFree(rgpu);
    cudaFree(bgpu);
    cudaFree(Apgpu);
    cudaFree(pgpu);

    for (int i = 0; i < 2; i++)
    {
        S1 = cudaStreamDestroy(streams[i]); 
        if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Destruction"<<endl;
        }   
    }

    delete[] z0;
    delete[] r0;
    G_Solver->~AMG_GPU_solver();

}

//AMG-PCG 4 Implementation
void Solver_PCG_4(sp_matrix_mg& A,double* &b,double* &x)
{
    cudaError_t S1;
    cudaStream_t *streams;

    streams = (cudaStream_t *) malloc(2 * sizeof(cudaStream_t));

    for (int i = 0; i < 2; i++)
    {
        S1 = cudaStreamCreate(&streams[i]);  
        if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Creation"<<endl;
        }  
    }

    // Preconditioner
    AMG_GPU1_solver* G_Solver = new AMG_GPU1_solver();
	G_Solver->AMG_solver_setup_jacobi(A);
	G_Solver->GPU_Allocations();

    sp_matrix_gpu *A1 = new sp_matrix_gpu(A);
    A1->matrix_transfer_gpu(A,streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

    double *Apgpu;
    double *pgpu;

    double *xgpu;
    double *bgpu;
    double *rgpu;

    double *zgpu;

    S1 = cudaMalloc(&Apgpu,(A.nrow)*sizeof(double));
    S1 = cudaMalloc(&pgpu,(A.nrow)*sizeof(double));
    S1 = cudaMalloc(&xgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&bgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&rgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&zgpu, (A.nrow)*sizeof(double));

    cudaMemcpy(xgpu,x,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(bgpu,b,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);

    double beta;
    double alpha;
    double alpha1;

    int count = 0;

    double s1 = 1.0;
    double s3 = 0.0;

    A1->cusparse_stat  = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat    = cublasSetStream  (A1->cublas_hndl,  streams[0]);
    A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,xgpu,&s3,rgpu);

    daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,bgpu,-1,rgpu);
   
    thrust::fill_n(thrust::device,zgpu,A.nrow,0.0); 

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    G_Solver->AMG_Solve(rgpu,zgpu,1);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    
    cudaMemcpy(pgpu,zgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);

    double r1;
    cublasDnrm2(A1->cublas_hndl,A.nrow,rgpu,1,&r1);
    
    count = 0;

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    while(r1 > tol1)
    {
        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,pgpu,&s3,Apgpu);

        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,pgpu,1,Apgpu,1,&alpha);
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,rgpu,1,zgpu, 1,&alpha1);

        alpha = alpha1/alpha;

        A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A.nrow,&alpha,pgpu,1,xgpu,1);
        alpha = -alpha;
        A1->cublas_stat   = cublasDaxpy(A1->cublas_hndl,A.nrow,&alpha,Apgpu,1,rgpu,1);
        
        thrust::fill_n(thrust::device,zgpu,A.nrow,0.0);
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
       
        G_Solver->AMG_Solve(rgpu,zgpu,1);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,rgpu,1,zgpu,1 ,&beta);
        beta = beta/alpha1;

        daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,zgpu,beta,pgpu);
        count++;
        cublasDnrm2(A1->cublas_hndl,A.nrow,rgpu,1,&r1);

        if(print_solve_phase_details == 1)
        {
            std::cout<<count<<"\t"<<r1<<"\n";
        }	
        
    }
    
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
    
    cudaMemcpy(x,xgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    A1->~sp_matrix_gpu();
    
    cudaDeviceSynchronize();

    G_Solver->~AMG_GPU1_solver();
    cudaDeviceSynchronize();
    
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    cudaFree(xgpu);
    cudaFree(Apgpu);
    cudaFree(pgpu);
    cudaFree(bgpu);
    cudaFree(rgpu);
    cudaFree(zgpu);
    cudaDeviceSynchronize();
}

//AMG-PBiCG 3 Implementation
void Solver_PBiCG_3(sp_matrix_mg& A,double* &b,double* &x)
{
    //Memory Allocations
    double *r0gpu;
    double *rgpu;
    double *pgpu;
    double *Apgpu;
    double *sgpu;
    double *Asgpu;
    double *p1gpu;
    double *s1gpu;
    
    cudaError_t S1;
    S1 = cudaMalloc(&r0gpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&rgpu,  (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&pgpu,  (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&Apgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&sgpu,  (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&Asgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&p1gpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&s1gpu, (A.nrow)*sizeof(double));
    
    //Constants
    double beta    = 0;
    double alpha   = 0;
    double alpha1  = 0;
    double omega1  = 0;

    double s1 = 1.0;
    double s3 = 0.0;

    double *xgpu;
    double *bgpu;
    
    S1 = cudaMalloc(&xgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&bgpu, (A.nrow)*sizeof(double));
    
    //Setup AMG Preconditioner
    AMG_GPU_solver* G_Solver = new AMG_GPU_solver();
	G_Solver->AMG_solver_setup_jacobi(A);
	G_Solver->GPU_Allocations();
    
    cudaStream_t *streams;
    streams = (cudaStream_t *) malloc(2 * sizeof(cudaStream_t));

    for (int i = 0; i < 2; i++)
    {
        S1 = cudaStreamCreate(&streams[i]); 
        if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Creation"<<endl;
        }   
    }

    sp_matrix_gpu *A1 = new sp_matrix_gpu(A);
    A1->matrix_transfer_gpu(A,streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();

    cudaMemcpy(xgpu,x,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(bgpu,b,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaDeviceSynchronize();
    
    //Compute Residual
    double* r0 = new double[A.nrow]();
    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r0);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,r0,1);
    
    //Initialization
    cudaMemcpy(rgpu, r0,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(r0gpu,r0,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(pgpu, r0,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    
    double r1 = cblas_dnrm2(A.nrow,r0,1);
    int count = 0;

    A1->cusparse_stat = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat   = cublasSetStream  (A1->cublas_hndl,  streams[0]);

    count = 0;
    cudaDeviceSynchronize();
   
    while(r1 > tol1)
    {
        thrust::fill_n(thrust::device,p1gpu,A.nrow,0.0);
        G_Solver->AMG_GPU_solve_1(pgpu,p1gpu,1);
        
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
   
        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,A1->descrA,A1->val,A1->rowptr,A1->colindex,p1gpu,&s3,Apgpu);
        
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
    
        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,r0gpu,1,Apgpu,1,&alpha);
        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,r0gpu,1,rgpu, 1,&alpha1);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        
        alpha = alpha1/alpha;

        daxpbyc<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,rgpu,-alpha,Apgpu,0.0,sgpu);
        
        thrust::fill_n(thrust::device,s1gpu,A.nrow,0.0);
                
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        
        G_Solver->AMG_GPU_solve_1(sgpu,s1gpu,1);
        
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        
        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,A1->descrA,A1->val,A1->rowptr,A1->colindex,s1gpu,&s3,Asgpu);

        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,Asgpu,1,sgpu,1 ,&omega1);
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,Asgpu,1,Asgpu,1,&beta);
        
        omega1 = omega1/beta;

        daxpbyc<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,alpha,p1gpu,omega1,s1gpu,1.0,xgpu);
        
        daxpbyc<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,sgpu,(-1*omega1),Asgpu,0.0,rgpu);
        
        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,r0gpu,1,rgpu,1,&beta);
        
        beta = beta*(alpha/omega1);
        
        daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,rgpu,beta,pgpu);
        daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,(-1*beta*omega1),Apgpu,1.0,pgpu);
        
        A1->cublas_stat = cublasDnrm2(A1->cublas_hndl,A.nrow,rgpu,1,&r1);
        if(print_solve_phase_details == 1)
        {
            cout<<count++<<"\t"<<r1<<endl;
        }
        
    }
    
    cudaMemcpy(x,xgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(rgpu);
    cudaFree(bgpu);
    cudaFree(Apgpu);
    cudaFree(pgpu);

    for(int i = 0; i < 2; i++)
    {
        S1 = cudaStreamDestroy(streams[i]); 
        if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Destruction"<<endl;
        }   
    }

    G_Solver->~AMG_GPU_solver();

}

//AMG-PBiCG 4 Implementation
void Solver_PBiCG_4(sp_matrix_mg& A,double* &b,double* &x)
{
    cudaError_t S1;
    cudaStream_t *streams;

    streams = (cudaStream_t *) malloc(2 * sizeof(cudaStream_t));

    for (int i = 0; i < 2; i++)
    {
        S1 = cudaStreamCreate(&streams[i]);  
        if(S1 != cudaSuccess)
        {
            cout<<"Error in Stream Creation"<<endl;
        }  
    }

    sp_matrix_gpu *A1 = new sp_matrix_gpu(A);
    A1->matrix_transfer_gpu(A,streams[0]);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    double *xgpu;
    double *bgpu;

    S1 = cudaMalloc(&xgpu, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&bgpu, (A.nrow)*sizeof(double));

    cudaMemcpy(xgpu,x,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(bgpu,b,((A).nrow)*sizeof(double),cudaMemcpyHostToDevice);


    //BICG required Allocations

    double* r0;
	double* r;
	double* p;
	double* Ap;
	double* s;
	double* As;
    double* p1;
    double* s0;

    //constants
    double alpha;
    double alpha1;
    double omega2;
    double omega1;
    double beta;

    double s1 = 1.0;
    double s3 = 0.0;

    S1 = cudaMalloc(&r , (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&r0, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&p,  (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&Ap, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&s,  (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&As, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&p1, (A.nrow)*sizeof(double));
    S1 = cudaMalloc(&s0, (A.nrow)*sizeof(double));

    //Initial Computations

    //Assign Streams
    A1->cusparse_stat  = cusparseSetStream(A1->cusparse_hndl,streams[0]);
    A1->cublas_stat    = cublasSetStream  (A1->cublas_hndl  ,streams[0]);

    A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,xgpu,&s3,r);

    daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,bgpu,-1,r);

    cudaMemcpy(r0,r,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(p ,r,((A).nrow)*sizeof(double),cudaMemcpyDeviceToDevice);

    double res;
    cublasDnrm2(A1->cublas_hndl,A.nrow,r,1,&res);

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    //Preconditioner
    AMG_GPU1_solver* G_Solver = new AMG_GPU1_solver();
	G_Solver->AMG_solver_setup_jacobi(A);
	G_Solver->GPU_Allocations();

    thrust::fill_n(thrust::device,p1,A.nrow,0.0);
    //G_Solver->AMG_GPU_solve_1(p,p1,1);
    
    int count = 0;
    while(res > tol1)
    {
        A1->cublas_stat   = cublasDdot(A1->cublas_hndl,A.nrow,r,1,r0,1 ,&alpha1);
        thrust::fill_n(thrust::device,p1,A.nrow,0.0);
        G_Solver->AMG_Solve(p,p1,1);
        
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,p1,&s3,Ap);

        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,Ap,1,r0,1 ,&alpha);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        alpha = alpha1/alpha;

        daxpbyc<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,r,-alpha,Ap,0.0,s);
        thrust::fill_n(thrust::device,s0,A.nrow,0.0);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        G_Solver->AMG_Solve(s,s0,1);    

        A1->cusparse_stat = cusparseDcsrmv(A1->cusparse_hndl,A1->transA,A1->nrow,A1->ncol,A1->nnz,&s1,
                                             A1->descrA,A1->val,A1->rowptr,A1->colindex,s0,&s3,As);

        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,As,1,s ,1,&omega2);
        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,As,1,As,1,&omega1);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        omega2 = omega2/omega1;
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        daxpbyc<<<A.nrow/thgpu + 1,thgpu,0,streams[1]>>>(A.nrow,alpha,p, omega,s ,1.0,xgpu);
        daxpbyc<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0  ,s,-omega,As,0.0,r);

        A1->cublas_stat = cublasDdot(A1->cublas_hndl,A.nrow,r,1,r0,1,&beta);
        
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        
        beta = (beta/alpha1)*(alpha/omega2);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);

        daxpby<<<A.nrow/thgpu + 1,thgpu,0,streams[0]>>>(A.nrow,1.0,r,beta,p);

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        beta = -1*beta*omega2;
        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
        A1->cublas_stat = cublasDaxpy(A1->cublas_hndl,A.nrow,&beta,Ap,1,p,1);

        cublasDnrm2(A1->cublas_hndl,A.nrow,r,1,&res);
        count++;

        if(print_solve_phase_details == 1)
        {
            cout<<count++<<"\t"<<res<<endl;
        }

        cudaStreamSynchronize(streams[0]);
        cudaStreamSynchronize(streams[1]);
    }

    cudaFree(r);
    cudaFree(r0);
    cudaFree(p);
    cudaFree(Ap);
    cudaFree(s);
    cudaFree(As);
    cudaFree(p1);
    cudaFree(s0);

    cudaMemcpy(x,xgpu,((A).nrow)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    G_Solver->~AMG_GPU1_solver();

    cudaFree(bgpu);
    A1->~sp_matrix_gpu();
    cudaFree(xgpu);
}
