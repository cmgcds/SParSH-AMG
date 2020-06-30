#ifndef AMG_GPU_PHASES_HPP_
#define AMG_GPU_PHASES_HPP_

#include "AMG_phases.hpp"
#include "AMG_gpu_matrix.hpp"

using namespace std;

//Solver Class for Hybrid AMG 1 Approach
class AMG_GPU_solver : public AMG_solver
{
    public:
        double* xgpu;
        double* bgpu;
        double* rgpu;
        double* hgpu;

        int    *P1rowgpu;
        double *P1valgpu;
        int    *P1colgpu;

        cudaStream_t *streams;
        int nstreams = 3;

        sp_matrix_gpu* A1; 
        sp_matrix_gpu* A2;


    public:
        using AMG_solver::AMG_solver;

        //Performs GPU Memory Allocations for Hybrid AMG 1 Approach
        void GPU_Allocations();

        //Solve Phase of AMG rhs and solution vector on CPU
        void AMG_GPU_solve(double* b,double* x,int iterations);

        //Solve Phase of AMG rhs and solution vector on GPU
        void AMG_GPU_solve_1(double* b,double* x,int iterations);

        //Helper functions for hybrid AMG 1 Approach
        

        //Method to overlap transfer of prolongator and smoothing iterations while going down the V cycle
        void smooth_and_get_prolongator(int k1);

        //Method to overlap restriction operations and getting next matrix on GPU
        void restrict_and_get_next(int k1);

        //Method to overlap smoothing iteration and transfer of prolongation matrix while going up the V cycle
        void smooth_and_prolongate(int k1);

        //Destructor to destroy solver object of hybrid AMG 1
        ~AMG_GPU_solver();
};

#endif