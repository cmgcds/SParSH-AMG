#ifndef AMG_SMOOTHERS_HPP_
#define AMG_SMOOTHERS_HPP_

#include "AMG_cpu_matrix.hpp"

namespace sequential
{
	void jacobi_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration = 2);

	void sor_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration = 2);
}

namespace parallel
{
	void jacobi_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration = 2);

	void sor_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration = 2);
}

#endif