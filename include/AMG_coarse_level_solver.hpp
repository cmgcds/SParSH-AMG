#ifndef AMG_COARSE_LEVEL_SOLVER_HPP_
#define AMG_COARSE_LEVEL_SOLVER_HPP_

#include "AMG_cpu_matrix.hpp"


class Direct_Solver_Pardiso
{
    public: 
        MKL_INT n;
        MKL_INT *ia  ;
        MKL_INT *ja  ;
        double *a    ;
        MKL_INT mtype;      
        MKL_INT nrhs ;     
        long int* pt;
        MKL_INT iparm[64];
        MKL_INT maxfct, mnum, phase, error, msglvl;
        MKL_INT i;
        double ddum;          
        MKL_INT idum; 

    public:

        Direct_Solver_Pardiso(sp_matrix_mg &A);
        void Direct_Solver_Pardiso_solve(double* &b,double* &x);
        ~Direct_Solver_Pardiso();

};

#endif