#include "AMG_smoothers.hpp"
#include "AMG.hpp"
#include "omp.h"

using namespace std;

namespace sequential
{
	void jacobi_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration)
	{
		mkl_set_num_threads(1);
		mkl_set_dynamic(false);

		int count = 0;
		while(count++ <= iteration)
		{
			A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,A.helper);
			cblas_daxpby(A.nrow,1.0,b,1,-1.0,A.helper,1);
			
			for(int i = 0; i < A.nrow;i++)
			{
				A.helper[i] /=A.diagonal[i];
			}

			cblas_daxpby(A.nrow,omega,A.helper,1,1.0,x,1);
		}

	}

	void sor_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration)
	{
		int count = 0;
		double sum;
		while(count++ < iteration)
		{
			for(int i = 0;i < A.nrow;i++)
			{
				sum = 0.0;
				for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
				{
					sum += A.val[j]*(x[A.colindex[j]]);
				}
				A.helper[i] = b[i] - sum;
				x[i] += omega*A.helper[i]/A.diagonal[i];
			}
		}

	}
}

namespace parallel
{
	void jacobi_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration)
	{
		mkl_set_num_threads(th);
		omp_set_num_threads(th);
		mkl_set_dynamic(false);

		int count = 0;
		while(count++ <= iteration)
		{
			A.sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A.A1,A.des,x,0.0,A.helper);
			cblas_daxpby(A.nrow,1.0,b,1,-1.0,A.helper,1);
			
			//For Debugging: Prints the residual
			//cout<<cblas_dnrm2((A).nrow,A.helper,1)<<endl;

			#pragma omp parallel for simd
				for(int i = 0; i < A.nrow;i++)
				{
					x[i] +=  omega*A.helper[i]/A.diagonal[i];
				}

			//cblas_daxpby(A.nrow,omega,A.helper,1,1.0,x,1);
		}
	}

	void sor_smoother(sp_matrix_mg& A,double* &b,double* &x,int iteration)
	{
		int count = 0;
		omp_set_num_threads(th);
		
		while(count++ < iteration)
		{ 
			for(int k = 0;k < A.total_colors;k++)
			{
				#pragma omp parallel for
                for(int l = A.color_count[k];l < A.color_count[k+1];l++)
                {
                    A.helper[l] = 0.0;
                    int lj;
                    for(lj = A.rowptr[l];lj < A.rowptr[l+1];lj++)
                    {
                        A.helper[l] += A.val[lj]*(x[A.colindex[lj]]);
                    }

                    A.helper[l] -= b[l];
                    x[l] -= omega*A.helper[l]/A.diagonal[l];
                }		
			}
		}
	}
}


