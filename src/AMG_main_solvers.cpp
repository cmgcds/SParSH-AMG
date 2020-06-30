#include "AMG.hpp"
#include "AMG_phases.hpp"
#include "AMG_gpu_phases.hpp"
#include "AMG_coarsening.hpp"
#include "AMG_gpu_phases_2.hpp"
#include "AMG_cycle_utilities.hpp"
#include "AMG_smoothers.hpp"
#include "iostream"
#include "omp.h"

using namespace std;

//AMG as solver with Jacobi as smoother
void AMG_Solver_1(sp_matrix_mg& A,double* &b,double* &x)
{
    AMG_solver* C_Solver   = new AMG_solver();
    double t1 = omp_get_wtime();
    C_Solver->AMG_solver_setup_jacobi(A);
    double t2 = omp_get_wtime();
    C_Solver->AMG_solve_jacobi(b,x,-1);
    double t3 = omp_get_wtime();
    cout<<"CPU Based Solver Setup Phase Time\t"<<t2-t1<<"\n";
    cout<<"CPU Based Solver Solve Phase Time\t"<<t3-t2<<"\n";
    cout<<"CPU Based Solver Total Time\t      "<<t3-t1<<"\n";
    C_Solver->~AMG_solver();
}


//AMG as solver with SOR as smoother
void AMG_Solver_2(sp_matrix_mg& A,double* &b,double* &x)
{
    AMG_solver* C_Solver   = new AMG_solver();
    double t1 = omp_get_wtime();
    C_Solver->AMG_solver_setup_SOR(A);
    parallel::reorder_rhs(A,b);
    double t2 = omp_get_wtime();
    C_Solver->AMG_solve_SOR(b,x,-1);
    parallel::reorder_rhs(A,b);
    parallel::reorder_rhs(A,x);
    double t3 = omp_get_wtime();
    cout<<t2-t1<<" "<<t3-t2<<" "<<t3-t1<<"\n";
    C_Solver->~AMG_solver();
}


//Conjugate Gradient Method
void Solver_CG_1(sp_matrix_mg& A,double* &b,double* &x)
{
    double *Ap       = new double[A.nrow];
    double *p        = new double[A.nrow];

    double beta;
    double alpha;

    double s;
    int count = 0;

    mkl_set_num_threads(th);
    mkl_set_dynamic(false);

    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,A.helper);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,A.helper,1);

    std::copy(b,b+A.nrow,A.helper);
    std::copy(A.helper,A.helper+A.nrow,p);
    double r1 =  cblas_dnrm2((A).nrow,A.helper,1);
    
    //double t1 = omp_get_wtime();

    while(count++ < A.nrow && r1 > 1e-8)
    {

        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,p,0.0,Ap);          //A*p_(n-1) = Ap

        alpha = cblas_ddot(A.nrow,p,1,Ap,1);													  //p_(n-1)TA*p_(n-1)
        s     = cblas_ddot(A.nrow,A.helper,1,A.helper,1);										  //r_(n-1)Tr_(n-1)
        alpha = s/alpha;

        cblas_daxpy(A.nrow,alpha,p,1,x,1);														  //update x
        cblas_daxpy(A.nrow,-alpha,Ap,1,A.helper,1);												  //r_n

        beta = cblas_ddot(A.nrow,A.helper,1,A.helper,1)/s;										  //r_nTr_n	
        cblas_daxpby(A.nrow,1.0,A.helper,1,beta,p,1);											  //pn = r_n + beta*p_(n-1)

        r1 = sqrt(s*beta);

        if(print_solve_phase_details == 1)
        {
            std::cout<<count<<"\t"<<r1<<"\n";  
        }
    }
    
    //double t3 = omp_get_wtime();
    //cout<<"Solving Time CG "<<t3-t1<<endl;
    
    if(count >= A.nrow)
    {
        cout<<"Conjugate Gradient Not Converged at Coarsest Level"<<endl;
    }

    delete[] Ap;
    delete[] p;
}


//AMG-PCG 1 CPU Implementation
void Solver_PCG_1(sp_matrix_mg& A,double* &b,double* &x)
{
    double *Ap       = new double[A.nrow];
    double *p        = new double[A.nrow];

    double* z0       = new double[A.nrow];
    double*	r0       = new double[A.nrow];

    double beta;
    double alpha;

    double s;
    int count = 0;
    
    mkl_set_num_threads(th);
    mkl_set_dynamic(false);

    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r0);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,r0,1);

    double r1 =  cblas_dnrm2((A).nrow,r0,1);

    AMG_solver* G_Solver   = new AMG_solver();

    G_Solver->AMG_solver_setup_jacobi(A);
    G_Solver->AMG_solve_jacobi(r0,z0,1);

    std::copy(z0,z0+A.nrow,p);

    while(count++ < A.nrow && r1 > tol1)
    {
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,p,0.0,Ap);

        alpha = cblas_ddot(A.nrow,p,1,Ap,1);
        s     = cblas_ddot(A.nrow,r0,1,z0,1);
        alpha = s/alpha;

        cblas_daxpy(A.nrow,alpha,p,1,x,1);
        cblas_daxpy(A.nrow,-alpha,Ap,1,r0,1);
        fill(z0,z0+A.nrow,0);
        G_Solver->AMG_solve_jacobi(r0,z0,1);

        beta = cblas_ddot(A.nrow,z0,1,r0,1)/s;
        cblas_daxpby(A.nrow,1.0,z0,1,beta,p,1);

        r1 = cblas_dnrm2((A).nrow,r0,1);
        
        if(print_solve_phase_details == 1)
        {
            std::cout<<count<<"\t"<<r1<<"\n";
        }	
        
    }

    delete[] Ap;
    delete[] p;
    delete[] r0;
    delete[] z0;

    G_Solver->~AMG_solver();
}


//AMG-PCG 2 Implementation
void Solver_PCG_2(sp_matrix_mg& A,double* &b,double* &x)
{
    double *Ap       = new double[A.nrow];
    double *p        = new double[A.nrow];

    double* z0       = new double[A.nrow];
    double*	r0       = new double[A.nrow];

    double beta;
    double alpha;

    double s;
    int count = 0;

    mkl_set_num_threads(th);
    mkl_set_dynamic(false);

    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r0);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,r0,1);

    double r1 =  cblas_dnrm2((A).nrow,r0,1);

    AMG_GPU_solver* G_Solver   = new AMG_GPU_solver();
    G_Solver->AMG_solver_setup_jacobi(A);
    G_Solver->GPU_Allocations();
    G_Solver->AMG_GPU_solve(r0,z0,1);

    std::copy(z0,z0+A.nrow,p);
    double t1 = omp_get_wtime();

    while(count++ < A.nrow && r1 > 1e-8)
    {
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,p,0.0,Ap);

        alpha = cblas_ddot(A.nrow,p,1,Ap,1);
        s     = cblas_ddot(A.nrow,r0,1,z0,1);
        alpha = s/alpha;

        cblas_daxpy(A.nrow,alpha,p,1,x,1);
        cblas_daxpy(A.nrow,-alpha,Ap,1,r0,1);
        fill(z0,z0+A.nrow,0);

        G_Solver->AMG_GPU_solve(r0,z0,1);
  
        beta = cblas_ddot(A.nrow,z0,1,r0,1)/s;
        cblas_daxpby(A.nrow,1.0,z0,1,beta,p,1);

        r1 = cblas_dnrm2((A).nrow,r0,1);

        if(print_solve_phase_details == 1)
        {
            std::cout<<count<<"\t"<<r1<<"\n";
        }	

    }

    double t2 = omp_get_wtime();
    cout<<"Solve Time PCG-2 "<<t2-t1<<endl;

    delete[] Ap;
    delete[] p;
    delete[] r0;
    delete[] z0;

    G_Solver->~AMG_GPU_solver();
}


//AMG as solver with Hybrid AMG 1 Approach
void AMG_Solver_3(sp_matrix_mg& A,double* &b,double* &x)
{
    AMG_GPU_solver* C_Solver   = new AMG_GPU_solver();
    double t1 = omp_get_wtime();
    C_Solver->AMG_solver_setup_jacobi(A);
    C_Solver->GPU_Allocations();
	double t2 = omp_get_wtime();
	C_Solver->AMG_GPU_solve(b,x,-1);
	double t3 = omp_get_wtime();
    cout<<"Time AMG Hybrid AMG 1\t"<<t3-t1<<endl; 	
	C_Solver->~AMG_GPU_solver();

}


//AMG as solver with Hybrid AMG 2 Approach
void AMG_Solver_4(sp_matrix_mg& A,double* &b,double* &x)
{
    AMG_GPU1_solver* C_Solver   = new AMG_GPU1_solver();
    double t1 = omp_get_wtime();
    C_Solver->AMG_solver_setup_jacobi(A);
    C_Solver->GPU_Allocations();
	double t2 = omp_get_wtime();
	C_Solver->helper(b,x,-1);
	double t3 = omp_get_wtime();
    cout<<"Time AMG Hybrid AMG 2\t"<<t3-t1<<endl; 	
	C_Solver->~AMG_GPU1_solver();

}

//BiCGStab as Solver
void Solver_BiCG_1(sp_matrix_mg& A,double* &b,double* &x)
{
    //Memory Allocations
    double* r0 = new double [A.nrow];
    double* r  = new double [A.nrow];
    double* p  = new double [A.nrow];
    double* Ap = new double [A.nrow];
    double* s  = new double [A.nrow];
    double* As = new double [A.nrow];

    //Constants
    double alpha  = 0;
    double alpha1 = 0;
    double omega1  = 0;
    double beta   = 0;

    //Set Threads
    mkl_set_num_threads(th);
    omp_set_num_threads(th);
    mkl_set_dynamic(false);

    //Compute Residual
    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r0);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,r0,1);	

    //Initialization
    std::copy(r0,r0+A.nrow,r);
    std::copy(r0,r0+A.nrow,p);

    double res = cblas_dnrm2(A.nrow,r0,1);
    int count = 0;
    while(res > tol1)
    {
        alpha1 = cblas_ddot(A.nrow,r,1,r0,1);
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,p,0.0,Ap);
        alpha = cblas_ddot(A.nrow,Ap,1,r0,1);

        alpha = alpha1/alpha;

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            s[i] = r[i] - alpha*Ap[i];
        }

        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,s,0.0,As);
        
        omega1 = cblas_ddot(A.nrow,As,1,s,1);
        omega1 /=cblas_ddot(A.nrow,As,1,As,1);

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            x[i] = x[i] + alpha*p[i] + omega1*s[i];
            r[i] = s[i] - omega1*As[i];
        }

        beta = cblas_ddot(A.nrow,r,1,r0,1)/alpha1;
        beta = beta*(alpha/omega1);

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            p[i] = r[i] + beta*(p[i] - omega1*Ap[i]);
        }

        res = cblas_dnrm2(A.nrow,r,1);

        if(print_solve_phase_details == 1)
        {
            cout<<count<<"\t"<<res<<endl;
        }
        

        count++;

    } 

    delete[] r;
    delete[] r0;
    delete[] p;
    delete[] Ap;
    delete[] s;
    delete[] As;
}

//AMG-PBiCG 1 CPU Implementation
void Solver_PBiCG_1(sp_matrix_mg& A,double* &b,double* &x)
{
    //Memory Allocations
    double* r0 = new double [A.nrow];
    double* r  = new double [A.nrow];
    double* p  = new double [A.nrow];
    double* Ap = new double [A.nrow];
    double* s  = new double [A.nrow];
    double* As = new double [A.nrow];

    double* p1 = new double [A.nrow];
    double* s1 = new double [A.nrow];

    //Constants
    double alpha  = 0;
    double alpha1 = 0;
    double omega1  = 0;
    double beta   = 0;

    //Set Threads
    mkl_set_num_threads(th);
    omp_set_num_threads(th);
    mkl_set_dynamic(false);

    //Compute Residual
    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r0);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,r0,1);	

    //Initialization
    std::copy(r0,r0+A.nrow,r);
    std::copy(r0,r0+A.nrow,p);

    double res = cblas_dnrm2(A.nrow,r0,1);
    int count = 0;

    //AMG Preconditioner Setup
    AMG_solver* G_Solver = new AMG_solver();
    G_Solver->AMG_solver_setup_jacobi(A);

    while(res > tol1)
    {
        std::fill(p1,p1+A.nrow,0.0);
        G_Solver->AMG_solve_jacobi(p,p1,1);

        alpha1 = cblas_ddot(A.nrow,r,1,r0,1);
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,p1,0.0,Ap);
        alpha = cblas_ddot(A.nrow,Ap,1,r0,1);

        alpha = alpha1/alpha;

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            s[i] = r[i] - alpha*Ap[i];
        }

        std::fill(s1,s1+A.nrow,0.0);
        G_Solver->AMG_solve_jacobi(s,s1,1);
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,s1,0.0,As);
        
        omega1  = cblas_ddot(A.nrow,As,1,s,1);
        omega1 /= cblas_ddot(A.nrow,As,1,As,1);

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            x[i] = x[i] + alpha*p1[i] + omega1*s1[i];
            r[i] = s[i] - omega1*As[i];
        }

        beta = cblas_ddot(A.nrow,r,1,r0,1)/alpha1;
        beta = beta*(alpha/omega1);

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            p[i] = r[i] + beta*(p[i] - omega1*Ap[i]);
        }

        res = cblas_dnrm2(A.nrow,r,1);
        
        if(print_solve_phase_details == 1)
        {
            cout<<count<<"\t"<<res<<endl;    
        }
        
        count++;

    }

    delete[] r;
    delete[] r0;
    delete[] p;
    delete[] Ap;
    delete[] s;
    delete[] As;
    delete[] s1;
    delete[] p1;

    G_Solver->~AMG_solver();
}

//AMG-PBiCG 2 Implementation
void Solver_PBiCG_2(sp_matrix_mg& A,double* &b,double* &x)
{
    //Memory Allocations
    double* r0 = new double [A.nrow];
    double* r  = new double [A.nrow];
    double* p  = new double [A.nrow];
    double* Ap = new double [A.nrow];
    double* s  = new double [A.nrow];
    double* As = new double [A.nrow];

    double* p1 = new double [A.nrow];
    double* s1 = new double [A.nrow];

    //Constants
    double alpha  = 0;
    double alpha1 = 0;
    double omega1  = 0;
    double beta   = 0;

    //Set Threads
    mkl_set_num_threads(th);
    omp_set_num_threads(th);
    mkl_set_dynamic(false);

    //Compute Residual
    A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r0);
    cblas_daxpby(A.nrow,1.0,b,1,-1.0,r0,1);	

    //Initialization
    std::copy(r0,r0+A.nrow,r);
    std::copy(r0,r0+A.nrow,p);

    double res = cblas_dnrm2(A.nrow,r0,1);
    int count = 0;

    //AMG Preconditioner Setup
    AMG_GPU_solver* G_Solver   = new AMG_GPU_solver();
    G_Solver->AMG_solver_setup_jacobi(A);
    G_Solver->GPU_Allocations();

    while(res > tol1)
    {
        
        std::fill(p1,p1+A.nrow,0.0);
        G_Solver->AMG_GPU_solve(p,p1,1);
        alpha1 = cblas_ddot(A.nrow,r,1,r0,1);
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,p1,0.0,Ap);
        alpha = cblas_ddot(A.nrow,Ap,1,r0,1);

        alpha = alpha1/alpha;

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            s[i] = r[i] - alpha*Ap[i];
        }
        
        std::fill(s1,s1+A.nrow,0.0);
        G_Solver->AMG_GPU_solve(s,s1,1);
       
        A.sA  = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,s1,0.0,As);
        
        omega1  = cblas_ddot(A.nrow,As,1,s,1);
        omega1 /= cblas_ddot(A.nrow,As,1,As,1);

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            x[i] = x[i] + alpha*p1[i] + omega1*s1[i];
            r[i] = s[i] - omega1*As[i];
        }

        beta = cblas_ddot(A.nrow,r,1,r0,1)/alpha1;
        beta = beta*(alpha/omega);

        #pragma omp parallel for simd
        for(int i = 0;i < A.nrow;i++)
        {
            p[i] = r[i] + beta*(p[i] - omega1*Ap[i]);
        }

        res = cblas_dnrm2(A.nrow,r,1);
       
        if(print_solve_phase_details == 1)
        {
            cout<<count<<"\t"<<res<<endl;
        }
        count++;
        
    }

    delete[] r;
    delete[] r0;
    delete[] p;
    delete[] Ap;
    delete[] s;
    delete[] As;
    delete[] s1;
    delete[] p1;

    G_Solver->~AMG_GPU_solver();
}


//Functions to test implementation
void coarsening_2(sp_matrix_mg& A,double* &b,double* &x)
{
    //sp_matrix_mg* P;
    int count = 0;
    A.color_matrix_and_reorder();
    while(count++ < 100)
    {
        parallel::sor_smoother(A,b,x,1);
        cout<<count<<"\t"<<parallel::residual(A,b,x)<<endl;
    }
    
    //sequential::HEM_Prolongator(A,P,0);
    //P->check_sp_matrix();
    //cout<<P->ncol<<endl;
}
