/*
 * cycle_utilities.cpp
 *
 *  Created on: 12-Dec-2019
 *      Author: gauss
 */

#include "AMG_cycle_utilities.hpp"
#include "AMG.hpp"
#include "omp.h"
#include "iostream"
#include "AMG_coarsening.hpp"

using namespace std;

namespace sequential
{
	//Computes Residual: L2 norm of Ax - b
	double residual(sp_matrix_mg& A,double* &b,double* &x)
	{
		mkl_set_num_threads(1);
		mkl_set_dynamic(false);
		A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,A.helper);

		cblas_daxpy(A.nrow,-1.0,b,1.0,A.helper,1);

		double res = cblas_dnrm2(A.nrow,A.helper,1);
		return res;
	}

	//Computes b = P^T*r
	void transfer_residual(sp_matrix_mg &P1,double* &r,double* &b)
	{
		mkl_set_num_threads(1);
		mkl_set_dynamic(false);
		P1.sA = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,1.0,P1.A1,P1.des,r,0.0,b);
	}

	//Computes x1 = x1 + P1*x
	void transfer_solution(sp_matrix_mg &P1,double* &x,double* &x1)
	{
		mkl_set_num_threads(1);
		mkl_set_dynamic(false);
		P1.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,P1.A1,P1.des,x,1.0,x1);
		
	}

	//Computes r = b - Ax
	void store_residual(sp_matrix_mg &A,double* &b,double* &x,double* &r)
	{
		mkl_set_num_threads(1);
		mkl_set_dynamic(false);
		A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r);
		cblas_daxpby(A.nrow,1.0,b,1,-1.0,r,1);

	}

	//Computes Ac = P^TAP
	void coarsen_matrix(sp_matrix_mg &A,sp_matrix_mg *&Ac,sp_matrix_mg &P1)
	{
		sparse_matrix_t C1;
		sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
		Ac = new sp_matrix_mg();
		mkl_set_num_threads(1);
		mkl_set_dynamic(false);

		A.sA = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,A.A1,P1.A1,&C1);
		
		A.sA = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE,P1.A1,C1,&(Ac->A1));

        A.sA = mkl_sparse_order(Ac->A1);
		A.sA = mkl_sparse_d_export_csr(Ac->A1,&p1,&(Ac->nrow),&(Ac->ncol),&(Ac->rowptr),&(Ac->rowptr) + 1,&(Ac->colindex),&(Ac->val));

		Ac->nnz = Ac->rowptr[Ac->nrow];

		mkl_sparse_destroy(C1);
	}
}

namespace parallel
{
	//Computes Residual: L2 norm of Ax - b
	double residual(sp_matrix_mg& A,double* &b,double* &x)
	{

		mkl_set_num_threads(th);
		mkl_set_dynamic(false);
		A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,A.helper);

		cblas_daxpy(A.nrow,-1.0,b,1.0,A.helper,1);

		double res = cblas_dnrm2(A.nrow,A.helper,1);
		return res;
	}

	//Computes b = P^T*r
	void transfer_residual(sp_matrix_mg &P1,double* &r,double* &b)
	{

		mkl_set_num_threads(th);
		mkl_set_dynamic(false);
		P1.sA = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,1.0,P1.A1,P1.des,r,0.0,b);

	}

	//Computes x1 = x1 + P1*x
	void transfer_solution(sp_matrix_mg &P1,double* &x,double* &x1)
	{
		mkl_set_num_threads(th);
		mkl_set_dynamic(false);
		P1.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,P1.A1,P1.des,x,1.0,x1);
	}

	//Computes r = b - Ax
	void store_residual(sp_matrix_mg &A,double* &b,double* &x,double* &r)
	{
		mkl_set_num_threads(th);
		mkl_set_dynamic(false);

		A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,r);
		cblas_daxpby(A.nrow,1.0,b,1,-1.0,r,1);

	}

	//Computes Ac = P^TAP
	void coarsen_matrix(sp_matrix_mg &A,sp_matrix_mg *&Ac,sp_matrix_mg &P1)
	{
		sparse_matrix_t C1;
		sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
		Ac = new sp_matrix_mg();
		mkl_set_num_threads(th);
		mkl_set_dynamic(false);

		A.sA = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,A.A1,P1.A1,&C1);
		A.sA = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE,P1.A1,C1,&(Ac->A1));
		
		A.sA = mkl_sparse_order(Ac->A1);
		A.sA = mkl_sparse_d_export_csr(Ac->A1,&p1,&(Ac->nrow),&(Ac->ncol),&(Ac->rowptr),&(Ac->rowptr) + 1,&(Ac->colindex),&(Ac->val));

		Ac->nnz = Ac->rowptr[Ac->nrow];

		mkl_sparse_destroy(C1);

		Ac->sp_matrix_fill_diagonal();

	}

	//Reorders Prolongators as per Colors of A
	void reorder_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P)
	{
		// Reordering the Matrix
		mkl_set_num_threads(th);
		int    *Prow = new int[A.nrow+1];
		double *Pval = new double[A.nrow];
		Prow[A.nrow] = A.nrow;

		for(int i = 0;i < A.nrow;i++)
		{
			Prow[i] = i;
			Pval[i] = 1.0;    
		}

		sparse_matrix_t Pt;
		sparse_status_t sA;
		sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;

		sA = mkl_sparse_d_create_csr(&Pt,SPARSE_INDEX_BASE_ZERO,
													A.nrow,A.ncol,Prow,Prow+1,A.color,Pval);
		if(sA != SPARSE_STATUS_SUCCESS)
		{
			std::cout<< "Matrix is not created" <<std::endl;
		}

		sA = mkl_sparse_order(Pt);

		sparse_matrix_t C1;
    	sA = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE,A.des,P->A1,SPARSE_OPERATION_TRANSPOSE,A.des,Pt,SPARSE_STAGE_FULL_MULT,&C1);
		sA = mkl_sparse_order(C1);
    	sA = mkl_sparse_d_export_csr(C1,&p1,&(P->nrow),&(P->ncol),&(P->rowptr),&(P->rowptr) + 1,&(P->colindex),&(P->val));

		P->sp_matrix_fill();

		delete[] Prow;
		delete[] Pval;
		


	}

	// Reorders RHS
	void reorder_rhs(sp_matrix_mg &A,double* &b)
	{
		// Reordering the Matrix
		mkl_set_num_threads(th);
		int    *Prow = new int[A.nrow+1];
		double *Pval = new double[A.nrow];
		Prow[A.nrow] = A.nrow;

		for(int i = 0;i < A.nrow;i++)
		{
			Prow[i] = i;
			Pval[i] = 1.0;    
		}

		sparse_matrix_t Pt;
		sparse_status_t sA;
		sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;

		sA = mkl_sparse_d_create_csr(&Pt,SPARSE_INDEX_BASE_ZERO,
													A.nrow,A.ncol,Prow,Prow+1,A.color,Pval);
		if(sA != SPARSE_STATUS_SUCCESS)
		{
			std::cout<< "Matrix is not created" <<std::endl;
		}

		sA = mkl_sparse_order(Pt);
		double* x1 = new double[A.nrow];

		sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,Pt,A.des,b,0.0,x1);
		copy(x1,x1+A.nrow,b);

		delete[] x1;
	}
	
}


	
