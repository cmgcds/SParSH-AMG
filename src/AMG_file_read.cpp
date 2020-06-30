/*
 * file_read_utilities.cpp
 *
 *  Created on: 11-Dec-2019
 *      Author: gauss
 */

#include "AMG.hpp"
#include "AMG_cpu_matrix.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

struct file_val
{
	int r;
	int c;
	double val;
};

int compare_function(const void* a1,const void* b1)
{
	file_val* a = (file_val*)a1;
	file_val* b = (file_val*)b1;

	if(a->r != b->r)
	{
		return (a->r > b->r);
	}
	else
	{
		return (a->c > b->c);
	}
	return 0;
}

void readcoo(char *matrixfile,char *rhsfile,sp_matrix_mg *&A,double *&b)
{
	ifstream in1;
	in1.open(matrixfile);
	int nrow;
	int ncol;
	int nnnz;
	in1>>nrow;
	in1>>ncol;
	in1>>nnnz;

	int k;

	A = new sp_matrix_mg(nrow,ncol,nnnz);
	b = new double[nrow];

	for(int i = 0;i < nnnz;i++)
	{
		in1>>k;
		A->rowptr[k+1]++;
		in1>>A->colindex[i];
		in1>>A->val[i];
	}

	in1.close();
	in1.open(rhsfile);
	in1>>k;
	for(int i = 0;i < nrow;i++)
	{
		in1>>b[i];
		A->rowptr[i+1] += A->rowptr[i];
	}

}

void read_coo_new_format(char *matrixfile,sp_matrix_mg *&A,double *&b)
{
	ifstream in1;
	in1.open(matrixfile);

	string matrix_details;
	getline(in1,matrix_details);
	
	int is_matrix     = 0;
	int is_real       = 0;
	int is_symmetric  = 0;
	int is_coordinate = 0;
	int is_general    = 0;

	string S1 = "matrix";
	string S2 = "coordinate";
	string S3 = "real";
	string S4 = "symmetric";
	string S5 = "general";

	//Get Basic Matrix Details
	istringstream det (matrix_details);
	string get_word;
	
	do
	{
		det >> get_word;

		if(S1.compare(get_word)==0)
		{
			is_matrix = 1;
			//cout<<"matrix "<<is_matrix<<endl;
			continue;
		}
		
		if(S2.compare(get_word)==0)
		{
			is_coordinate = 1;
			//cout<<"coordinate "<<is_coordinate<<endl;
			continue;
		}

		if(S3.compare(get_word)==0)
		{
			is_real = 1;
			//cout<<"real "<<is_real<<endl;
			continue;
		}

		if(S4.compare(get_word)==0)
		{
			is_symmetric = 1;
			//cout<<"symmetric "<<is_symmetric<<endl;
			continue;
		}

		if(S5.compare(get_word)==0)
		{
			is_general = 1;
			//cout<<"general "<<is_general<<endl;
			continue;
		}

	} while (det);


	int lines_to_skip = 1;
	string test;
	
	getline(in1,test);
	
	while(test[0]=='%')
	{
		lines_to_skip++;
		getline(in1,test);
	}

	in1.close();
	in1.open(matrixfile);
	
	while(lines_to_skip > 0)
	{
		getline(in1,test);
		lines_to_skip--;
	}

	int nrow;
	int ncol;
	int nnnz;
	int k;
	
	in1>>nrow;
	in1>>ncol;
	in1>>nnnz;

	A = new sp_matrix_mg(nrow,ncol,nnnz);
	b = new double[nrow]();

	for(int i = 0;i < nnnz;i++)
	{
		in1>>k;
		A->rowptr[k+1]++;
		in1>>A->colindex[i];
		in1>>A->val[i];
	}

	for(int i = 0;i < nrow;i++)
	{
		in1>>b[i];
		A->rowptr[i+1] += A->rowptr[i];
	}
}

