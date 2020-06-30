#ifndef AMG_PHASES_HPP_
#define AMG_PHASES_HPP_

#include "AMG_cpu_matrix.hpp"
#include "AMG_coarse_level_solver.hpp"
#include "AMG.hpp"

class AMG_solver
{
    public:

        //Number of Levels used in AMG
        int l;
        
        //Hierarchy of A Matrix
        sp_matrix_mg **Av = new sp_matrix_mg*[level1];
		
        //Hierarchy of P Matrix
        sp_matrix_mg **Pv = new sp_matrix_mg*[level1-1];

        //Stack of solution(X), RHS and Residual vector
		double **Xv = new double*[level1];
		double **Bv = new double*[level1];
		double **Rv = new double*[level1];

        //Coarse Level Solver
        Direct_Solver_Pardiso* Directsolve;


    public:

        //Constructor to create AMG solver object
        AMG_solver();

        //Method to setup hierarchy of matrices for jacobi as smoother
        void AMG_solver_setup_jacobi(sp_matrix_mg &A);
         
         //Method to setup hierarchy of matrices for SOR as smoother         
        void AMG_solver_setup_SOR(sp_matrix_mg &A);

        //Method to solve phase of AMG with Jacobi as smoother
        void AMG_solve_jacobi(double* &b,double* &x,int iterations);

        //Method to solve phase of AMG with SOR as smoother
        void AMG_solve_SOR(double* &b,double* &x,int iterations);

        //void AMG_solve_jacobi_GPU(double* &b,double* &x,int iterations);

        //void AMG_solve_SOR_GPU(double* &b,double* &x,int iterations);

        //Destructor to destroy AMG solver object
        ~AMG_solver();
};

#endif