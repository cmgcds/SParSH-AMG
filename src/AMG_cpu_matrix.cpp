/*
 * AMG_cpu.cpp
 *
 *  Created on: 11-Dec-2019
 *      Author: gauss
 */
#include "AMG.hpp"
#include "AMG_cpu_matrix.hpp"
#include<iostream>
#include<vector>
#include<algorithm>
#include<cmath>

using namespace std;

//Function creates matrix for mkl_sparse_matrix datatype
void sp_matrix_mg::sp_matrix_fill()
{
	sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;

	sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
												 nrow,ncol,rowptr,rowptr+1,colindex,val);
	if(sA != SPARSE_STATUS_SUCCESS)
	{
		std::cout<< "Matrix is not created" <<std::endl;
	}

	sA = mkl_sparse_order(A1);
	des.type = SPARSE_MATRIX_TYPE_GENERAL;
}


//Function extracts diagonal entries from matrix and stores it in diagonal array
void sp_matrix_mg::sp_matrix_fill_diagonal()
{
	des.type = SPARSE_MATRIX_TYPE_GENERAL;
	diagonal = new double[nrow];
	helper   = new double[nrow];
	for(int i = 0;i < nrow;i++)
	{
		for(int j = rowptr[i];j < rowptr[i+1];j++)
		{
			if(colindex[j]==i)
			{
				diagonal[i] = val[j];
				break;
			}
		}
	}
}


//Destructor to destroy sp_matrix_mg object
sp_matrix_mg::~sp_matrix_mg()
{
	//cout<<"Derived Class Destructor Called"<<endl;
	mkl_sparse_destroy(A1);

	if(nrow == ncol)
	{
		delete[] diagonal;
		delete[] helper;
		delete[] entries;
	}

    if(color != nullptr)
    {
        delete[] color;
    }

    if(color_count != nullptr)
    {
        delete[] color_count;
    }

}


//Function performs coloring and reordering of matrix - complete implementation is pending
void sp_matrix_mg::color_matrix_and_reorder()
{
	color = new int[nrow];
	fill_n(color,nrow,0);
	
	//Find upper limit on number of colors;
	int max_degree = 0;
	for(int i = 0;i < nrow;i++)
	{
		color[i] = rowptr[i+1] - rowptr[i];
	}

	int max_count = *max_element(color,color+nrow);

	fill_n(color,nrow,0);

	int *forbidden = new int[max_count+1];
	color_count    = new int[max_count+1];

	fill_n(color_count,max_count+1,0);

	for(int i = 0;i < nrow;i++)
	{
		fill_n(forbidden,max_count+1,-1);
		for(int j = rowptr[i];j < rowptr[i+1];j++)
		{
			if(color[colindex[j]]!=0)
			{
				forbidden[color[colindex[j]]] = i;
			}	
		}

		int c = INT32_MAX;
		for(int k = 1;k < max_count+1;k++)
		{
			if(forbidden[k]!=i)
			{
				c = min(c,k);
				break;
			}
		}

		color[i] = c;
		color_count[c]++;
		total_colors = max(total_colors,c);
	
	}

	vector<vector<int> >A2(total_colors+1);
	for(int i = 0;i < nrow;i++)
	{
		A2[color[i]].push_back(i);
	}

	cout<<endl;
 
	for(int i = 0;i < total_colors;i++)
	{
		color_count[i+1] += color_count[i];
		copy(A2[i+1].begin(),A2[i+1].end(),color+color_count[i]);  
	}

	// Reordering the Matrix
	mkl_set_num_threads(th);
	
	int    *Prow = new int[nrow+1];
    double *Pval = new double[nrow];
    Prow[nrow] = nrow;

    for(int i = 0;i < nrow;i++)
    {
        Prow[i] = i;
        Pval[i] = 1.0;    
    }

	sparse_matrix_t Pt;

    sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;

	sA = mkl_sparse_d_create_csr(&Pt,SPARSE_INDEX_BASE_ZERO,nrow,ncol,Prow,Prow+1,color,Pval);
	if(sA != SPARSE_STATUS_SUCCESS)
	{
		std::cout<< "Matrix is not created" <<std::endl;
	}

	sA = mkl_sparse_order(Pt);
	sparse_matrix_t C1;
	des.type = SPARSE_MATRIX_TYPE_GENERAL;

    sA = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE,des,Pt,SPARSE_OPERATION_NON_TRANSPOSE,des,A1,SPARSE_STAGE_FULL_MULT,&C1);

    if(sA != SPARSE_STATUS_SUCCESS)
    {
        cout<<"Error in 1" <<endl;
		cout<<sA<<endl;
    }

	sparse_matrix_t C2;
    sA = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE,des,C1,SPARSE_OPERATION_TRANSPOSE,des,Pt,SPARSE_STAGE_FULL_MULT,&C2);
    if(sA != SPARSE_STATUS_SUCCESS)
    {
        cout<<"Error in 2"<<endl;
    }

	sA = mkl_sparse_order(C2);
    sA = mkl_sparse_d_export_csr(C2,&p1,&(nrow),&(ncol),&(rowptr),&(rowptr) + 1,&(colindex),&(val));

	 if(sA != SPARSE_STATUS_SUCCESS)
    {
        cout<<"Error in 3"<<endl;
    }

    sp_matrix_fill();
    sp_matrix_fill_diagonal();   

	delete[] Prow;
	delete[] Pval;
}


//Function which divides each row by its diagonal entries of matrix
void sp_matrix_mg::normalize_matrix()
{
	double* norm1 = new double[ncol]();
	for(int i = 0;i < nrow;i++)
    {	
        for(int j = rowptr[i];j < rowptr[i+1];j++)
        {
			norm1[colindex[j]]  += val[j]*val[j]; 
		}
	}

	for(int i = 0;i < nrow;i++)
    {	
        for(int j = rowptr[i];j < rowptr[i+1];j++)
        {
			val[j] = val[j]/norm1[colindex[j]]; 
		}
	}

	delete[] norm1;
}


//Function scales vector b by diagonal entries of matrix
void sp_matrix_mg::scale_system(double* &b)
{
	for(int i = 0;i < nrow;i++)
    {	
        for(int j = rowptr[i];j < rowptr[i+1];j++)
        {
			val[j] = val[j]/diagonal[i];	
		}
		b[i] = b[i]/sqrt(diagonal[i]);
	}
}
