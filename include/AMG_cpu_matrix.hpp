#ifndef AMG_CPU_MATRIX_HPP_
#define AMG_CPU_MATRIX_HPP_

#include "AMG_matrix.hpp"

#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"


//Derived Class for storing Matrices in CSR Form with MKL Matrix Datatype
class sp_matrix_mg : public sp_matrix
{
	public:

		sparse_matrix_t A1;
		sparse_status_t sA;
		matrix_descr des;

		double *diagonal = nullptr;
		double *helper   = nullptr;
		double *entries  = nullptr;

		int *color       = nullptr;
		int *color_count = nullptr;
		int total_colors = 0;
		int max_color_row = 0;

	public:

		using sp_matrix::sp_matrix;
		//using sp_matrix::~sp_matrix;

		//Populate MKL Matrix from CSR matrix
		void sp_matrix_fill();

		//Populate diagonal array with diagonal entries of matrix
		void sp_matrix_fill_diagonal();

		//Color the matrix with reorders the matrix
		void color_matrix_and_reorder();
	
        //divides each row by its diagonal entries of matrix
		void scale_system(double* &b);

        //scales vector b by diagonal entries of matrix
		void normalize_matrix();

		//destructor for derived class
		~sp_matrix_mg();
};

#endif /* AMG_CPU_MATRIX_HPP_ */
