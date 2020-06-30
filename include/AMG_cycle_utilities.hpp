#ifndef AMG_CYCLE_UTILITIES_HPP_
#define AMG_CYCLE_UTILITIES_HPP_

#include "AMG_cpu_matrix.hpp"

namespace sequential
{
	double residual(sp_matrix_mg& A,double* &b,double* &x);

	void transfer_residual(sp_matrix_mg &P1,double* &r,double* &b);

	void transfer_solution(sp_matrix_mg &P1,double* &x,double* &x1);

	void store_residual(sp_matrix_mg &A,double* &b,double* &x,double* &r);

	void coarsen_matrix(sp_matrix_mg &A,sp_matrix_mg *&Ac,sp_matrix_mg &P1);
}

namespace parallel
{
	double residual(sp_matrix_mg& A,double* &b,double* &x);

	void transfer_residual(sp_matrix_mg &P1,double* &r,double* &b);

	void transfer_solution(sp_matrix_mg &P1,double* &x,double* &x1);

	void store_residual(sp_matrix_mg &A,double* &b,double* &x,double* &r);

	void coarsen_matrix(sp_matrix_mg &A,sp_matrix_mg *&Ac,sp_matrix_mg &P1);

	void reorder_rhs(sp_matrix_mg &A,double* &b);

	void reorder_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P);
}

#endif /* CYCLE_UTILITIES_HPP_ */