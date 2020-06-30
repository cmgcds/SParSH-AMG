#include "AMG_coarse_level_solver.hpp"
#include "stdio.h"
#include <iostream>
#include "AMG.hpp"
#include "omp.h"

using namespace std;

Direct_Solver_Pardiso::Direct_Solver_Pardiso(sp_matrix_mg &A)
{
    n   = A.nrow;
    ia  = A.rowptr;
    ja  = A.colindex;
    a   = A.val;

    pt = new long int[64];
    for ( i = 0; i < 64; i++ )
    {
        iparm[i] = 0;
        pt[i]    = 0; 
    }

    mtype = 11;
    nrhs  = 1;

    iparm[0] = 1;         /* No solver default */
    iparm[1] = 3;         /* Fill-in reordering from METIS */
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[7] = 0;         /* Max numbers of iterative refinement steps */
    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */
    iparm[34] = 1;        /* PARDISO use C-style indexing for ia and ja arrays */
    iparm[59] = 0;
    iparm[26] = 1;
    iparm[23] = 1;
    //iparm[33] = th;
    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;             /* Which factorization to use. */
    msglvl = 0;           /* Print statistical information in file */
    error = 0;            /* Initialize error flag */

    mkl_set_num_threads(th);
    kmp_set_warnings_off();
    phase = 12;
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,&n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf("\nERROR during symbolic factorization: %d", error);
        exit (1);
    }
    else
    {
        //cout<<"Numeric Factorization Computed and Stored"<<endl;
    }
}

void Direct_Solver_Pardiso::Direct_Solver_Pardiso_solve(double* &b,double* &x)
{
    phase = 33;
    mkl_set_num_threads(th);
    kmp_set_warnings_off();
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
                    &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during Back Substitution: %d", error);
        exit (1);
    }
}

Direct_Solver_Pardiso::~Direct_Solver_Pardiso()
{
    delete[] pt;
}
