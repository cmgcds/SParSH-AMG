#include "AMG_cpu_matrix.hpp"
#include "AMG_gpu_matrix.hpp"

//Utilties for Hybrid AMG 1 Approach

//Cycle Utilities Functions

//Function to pin the memory Locations for Matrix vector and rhs
void pin_AMG_MG_matrix(sp_matrix_mg& A,double* &b,double* &x,sp_matrix_mg& P);

//Function to unpin the memory Locations for Matrix vector and rhs
void unpin_AMG_MG_matrix(sp_matrix_mg& A,double* &b,double* &x,sp_matrix_mg& P);

//Function to swap pointers
void gpu_swap_pointers(sp_matrix_gpu* &A1,sp_matrix_gpu* &A2);

//Function to compute residual on GPU
double residual(sp_matrix_gpu &A,double* b,double* x,double* h,cudaStream_t streams);