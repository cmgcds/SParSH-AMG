/*
 * main.cpp
 *
 *  Created on: 11-Dec-2019
 *      Author: gauss
 */

#include "AMG.hpp"
#include <iostream>

using namespace std;

int main(int n,char *argv[])
{
    //Initialization
	sp_matrix_mg* A = new sp_matrix_mg();
	double *b;
	readcoo(argv[1],argv[2],A,b);
	double *x = new double[A->nrow]();
	std::fill(x,x+A->nrow,0);
	A->sp_matrix_fill();
	A->sp_matrix_fill_diagonal();
	
	cout<<"Matrix Size\t"<<A->nrow<<endl;
	AMG_Solver_CPU_GPU_CI(*A,b,x);

	std::fill(x,x+A->nrow,0);
	AMG_Solver_CPU_GPU_MI(*A,b,x);

    std::fill(x,x+A->nrow,0);
	AMG_Solver_CPU_baseline(*A,b,x);
    
    std::fill(x,x+A->nrow,0);
    //coarsening_2(*A,b,x);
    
	//std::fill(x,x+A->nrow,0);
	//AMG_Solver_4(*A,b,x);

	//Finalization
	A->~sp_matrix_mg();
   	delete[] x;
    delete[] b;
}


