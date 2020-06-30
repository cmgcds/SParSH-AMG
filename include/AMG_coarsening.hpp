#ifndef AMG_COARSENING_HPP_
#define AMG_COARSENING_HPP_

#include "AMG_cpu_matrix.hpp"

//These functions return Prolongation matrices obtained from specified coarsening strategies
namespace sequential
{
    //Heavy Edge Matching based Coarsening
	void HEM_Prolongator(sp_matrix_mg& A,sp_matrix_mg*  &P,int l1);
    
    //MIS based coarsening
    void mis_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P1);

    //Becks coarsening Algorithm
    void beck_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P1);

    //Compatible weighted matching based coarsening
    void C_W_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P,int l1);

    //Heuristic 2
    void HEM_Prolongator_2(sp_matrix_mg& A,sp_matrix_mg*  &P);
 
};

#endif