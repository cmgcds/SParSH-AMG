/*
        Main Header for package
        To be included in file in which functions from package are to be called
*/

#ifndef AMG_HPP_
#define AMG_HPP_

//Include Files which define matrix class and related functions
#include "AMG_matrix.hpp"
#include "AMG_cpu_matrix.hpp"

/* Parameters*/

#define th 2                           //Number of OpenMP Threads to be used in solver
#define omega 1.66667                  //Relaxation Factor in Jacobi and Gauss-Siedal Smoother
#define nsmooth 6                      //Number of smoothing iterations
#define tol1 1e-8                      //Tolerance of Solver                          
#define limit_upper 4000               //Upper Limit on coarse level system size
#define limit_lower 2000               //Lower Limit on coarse level system size
#define level1 6                       //Levels in AMG
#define smooth_iter 6                  //Smoothing iteration in each level

#define print_setup_phase_details 1    //Set it to 1 to print setup_phase_details
#define print_solve_phase_details 1    //Set it to 1 to print solve_phase_details 

#define thgpu 1024                     //Threads per block to be used in GPU implementation         

//Funcions to read matrix from file

//Read Matrix and RHS from 2 different files
void readcoo(char *matrixfile,char *rhsfile,sp_matrix_mg *&A,double *&b);

//Read Matrix and RHS from same file
void read_coo_new_format(char *matrixfile,sp_matrix_mg *&A,double *&b);

//Solvers

//AMG as solver with Jacobi as smoother
void AMG_Solver_1(sp_matrix_mg& A,double* &b,double* &x);

//AMG as solver with SOR as smoother
void AMG_Solver_2(sp_matrix_mg& A,double* &b,double* &x);

//CG as Solver
void Solver_CG_1(sp_matrix_mg& A,double* &b,double* &x);

//CG as Solver GPU Implementation
void Solver_CG_2(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PCG 1 CPU Implementation
void Solver_PCG_1(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PCG 2 Implementation
void Solver_PCG_2(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PCG 3 Implementation
void Solver_PCG_3(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PCG 4 Implementation
void Solver_PCG_4(sp_matrix_mg& A,double* &b,double* &x);

//AMG as solver with Hybrid AMG 1 Approach
void AMG_Solver_3(sp_matrix_mg& A,double* &b,double* &x);

//AMG as solver with Hybrid AMG 2 Approach
void AMG_Solver_4(sp_matrix_mg& A,double* &b,double* &x);

//BiCGStab as Solver
void Solver_BiCG_1(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PBiCG 1 CPU Implementation
void Solver_PBiCG_1(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PBiCG 2 Implementation
void Solver_PBiCG_2(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PBiCG 3 Implementation
void Solver_PBiCG_3(sp_matrix_mg& A,double* &b,double* &x);

//AMG-PBiCG 4 Implementation
void Solver_PBiCG_4(sp_matrix_mg& A,double* &b,double* &x);

//Functions to test implementation
void coarsening_2(sp_matrix_mg& A,double* &b,double* &x);

#endif
