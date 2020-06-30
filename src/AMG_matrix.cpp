/*
 * AMG.cpp
 *
 *  Created on: 11-Dec-2019
 *      Author: gauss
 */

#include "AMG.hpp"
#include "AMG_matrix.hpp"
#include<iostream>

using namespace std;

//Constructor to allocate memory for Matrix in CSR Form r X c matrix with n nonzeros
sp_matrix::sp_matrix(int r,int c,int n)
{
	nrow = r;
	ncol = c;
	nnz = n;

	rowptr    = new int[r+1];
	colindex  = new int[n];
	val       = new double[n];

	for(int i = 0; i < n;i++)
	{
		colindex[i] = 0;
		val[i] = 0.0;
	}

	for(int i = 0; i < r+1;i++)
	{
		rowptr[i] = 0;
	}

}

/*sp_matrix::~sp_matrix()
{
	cout<<"Base Class Destructor Called"<<endl;
	delete[] rowptr;
	delete[] colindex;
	delete[] val;
}*/

sp_matrix::sp_matrix(){};

//Prints matrix in CSR Form
void sp_matrix::check_sp_matrix()
{
	cout<<"\n Number of Rows: "<<nrow;
	cout<<"\n Number of columns "<<ncol;
	cout<<"\n Number of non-zeros "<<nnz;
	cout<<endl;

	for(int i = 0; i < nrow;i++)
	{
		cout<<i<<endl;
		for(int j = rowptr[i];j < rowptr[i+1];j++)
		{
			cout<<colindex[j]<<"\t";
			cout<<val[j]<<"\t";
			cout<<endl;
		}
		cout<<endl;
		cout<<endl;
	}
}