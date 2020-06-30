

#ifndef AMG_GPU_PHASES_2_HPP_
#define AMG_GPU_PHASES_2_HPP_

#include "AMG_phases.hpp"
#include "AMG_gpu_matrix.hpp"

using namespace std;

class AMG_GPU1_solver : public AMG_solver
{
    public:
        sp_matrix_gpu** Avgpu;
        sp_matrix_gpu** Pvgpu;

        double** Xvgpu;
        double** Bvgpu;
        double** Rvgpu;

        cudaStream_t *streams;
        int nstreams = 2;
        
        double* bgpu;
        double* xgpu;
    
    public:
        using AMG_solver::AMG_solver;

        
        void GPU_Allocations();

        //B and X on CPU
        void helper(double* b,double* x,int iterations);
        
        //B and X on GPUs
        void AMG_Solve(double* b,double* x,int iterations);

        
        ~AMG_GPU1_solver();

};

#endif
