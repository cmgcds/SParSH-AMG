#include "AMG_phases.hpp"
#include "AMG_smoothers.hpp"
#include "AMG_cycle_utilities.hpp"
#include "AMG_coarsening.hpp"
#include<algorithm>
#include<iostream>

using namespace std;

//Constructor to create AMG solver object
AMG_solver::AMG_solver(){};


//Destructor to destroy AMG solver object
AMG_solver::~AMG_solver()
{
    //Destruction of Matrices
	Directsolve->~Direct_Solver_Pardiso();
    for(int q = l;q > 0;q--)
    {
        delete[] Bv[q];
        delete[] Xv[q];
        Av[q]->~sp_matrix_mg();
        Pv[q-1]->~sp_matrix_mg();
    }

    delete[] Bv;
    delete[] Xv;
    delete[] Av;
    delete[] Pv;
};


//Method to setup hierarchy of matrices for jacobi as smoother
void AMG_solver::AMG_solver_setup_jacobi(sp_matrix_mg &A)
{
    l = 0;
    
	//Setup Phase
	Av[0] = &A;

	Xv[0] = new double[Av[0]->nrow];
	Bv[0] = new double[Av[0]->nrow];
	Rv[0] = new double[Av[0]->nrow];

    if(print_setup_phase_details == 1)
    {
        cout<<"AMG Setup Phase Details Jacobi smoother"<<endl;
    }

    while(Av[l]->nrow > limit_upper && l < (level1-1))
    {
        fill(Xv[l],Xv[l]+Av[l]->nrow,0.0);

        if(print_setup_phase_details == 1)
        {
            cout<<"Level "<<l<<":\t"<<Av[l]->nrow<<endl;
        }
       
        //Call the any one of below coarsening strategy
		sequential::HEM_Prolongator(*Av[l],Pv[l],l);
        //sequential::mis_prolongator(*Av[l],Pv[l]);
        //sequential::beck_prolongator(*Av[l],Pv[l]);
		//sequential::HEM_Prolongator_2(*Av[l],Pv[l]);
        //sequential::C_W_prolongator(*Av[l],Pv[l],l);
		
        //Compute PTAP product
        parallel::coarsen_matrix(*Av[l],Av[l+1],*Pv[l]);
        Av[l+1]->sp_matrix_fill_diagonal();
        
        //Allocate memory for X, rhs, and residual for next level
        l = l + 1;
        Xv[l] = new double [Av[l]->nrow]();
        Bv[l] = new double [Av[l]->nrow]();
        Rv[l] = new double [Av[l]->nrow]();

        if(Av[l]->nrow < limit_lower)
        {
            break;
        }
        
    }
   
    if(print_setup_phase_details == 1)
    {
        cout<<"Level "<<l<<":\t"<<Av[l]->nrow<<endl;
    }
	
    Directsolve = new Direct_Solver_Pardiso(*Av[l]);    
};


//Method to setup hierarchy of matrices for SOR as smoother
void AMG_solver::AMG_solver_setup_SOR(sp_matrix_mg &A)
{
	l = 0;
	Av[0] = &A;

	Xv[0] = new double[Av[0]->nrow];
	Bv[0] = new double[Av[0]->nrow];
	Rv[0] = new double[Av[0]->nrow];

    if(print_setup_phase_details == 1)
    {
        cout<<"AMG Setup Phase Details SOR Smoother"<<endl;
    }

	Av[l]->color_matrix_and_reorder();
	while(Av[l]->nrow > limit_upper && l < (level1-1))
	{
		//fill(Xv[l],Xv[l]+Av[l]->nrow,0.0);
        if(print_setup_phase_details == 1)
        {
            cout<<"Level "<<l<<":\t"<<Av[l]->nrow<<endl;
        }
		
        //Call the any one of below coarsening strategy
		sequential::HEM_Prolongator(*Av[l],Pv[l],l);
        //sequential::mis_prolongator(*Av[l],Pv[l]);
        //sequential::beck_prolongator(*Av[l],Pv[l]);
        //sequential::C_W_prolongator(*Av[l],Pv[l],l);

        
		parallel::coarsen_matrix(*Av[l],Av[l+1],*Pv[l]);
		Av[l+1]->color_matrix_and_reorder();
        Av[l+1]->sp_matrix_fill_diagonal();

		parallel::reorder_prolongator(*Av[l+1],Pv[l]);

		l = l + 1;
		Xv[l] = new double [Av[l]->nrow];
		Bv[l] = new double [Av[l]->nrow];
		Rv[l] = new double [Av[l]->nrow];

		if(Av[l]->nrow < limit_lower)
		{
			break;
		}	
	}

	if(print_setup_phase_details == 1)
    {
        cout<<"Level "<<l<<":\t"<<Av[l]->nrow<<endl;
    }

	Directsolve = new Direct_Solver_Pardiso(*Av[l]);    
};


//Method to solve phase of AMG with Jacobi as smoother
void AMG_solver::AMG_solve_jacobi(double* &b,double* &x,int iterations)
{
    int cycles = 0;
	double r1  = 0.0;

	copy(b,b+Av[0]->nrow,Bv[0]);
	copy(x,x+Av[0]->nrow,Xv[0]);
   
	r1 = parallel::residual(*Av[0],Bv[0],Xv[0]);
    
    int l1 = 0;

    if(iterations > 0)
    {
      	while(cycles < iterations)
		{
	 		while(l1 < l)
     		{
	 			parallel::jacobi_smoother(*Av[l1],Bv[l1],Xv[l1],smooth_iter);
	 			parallel::store_residual(*Av[l1],Bv[l1],Xv[l1],Rv[l1]);
	 			parallel::transfer_residual(*Pv[l1],Rv[l1],Bv[l1+1]);
	 			fill(Xv[l1+1],Xv[l1+1] + (Av[l1+1]->nrow),0.0);
	 			l1++;
	 		}
			
			//Direct Solve
			Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);

			l1 = l;

			while(l1 > 0)
			{
				parallel::transfer_solution(*Pv[l1-1],Xv[l1],Xv[l1-1]);
				parallel::jacobi_smoother(*Av[l1-1],Bv[l1-1],Xv[l1-1],smooth_iter);
				l1 = l1 - 1;
			}

			cycles++;
			r1 = parallel::residual(*Av[l1],Bv[l1],Xv[l1]);
			
		}
    }
	
	if(iterations == -1)
    {
        while(r1 > tol1)
		{
			while(l1 < l)
			{
				parallel::jacobi_smoother(*Av[l1],Bv[l1],Xv[l1],smooth_iter);
				parallel::store_residual(*Av[l1],Bv[l1],Xv[l1],Rv[l1]);
				parallel::transfer_residual(*Pv[l1],Rv[l1],Bv[l1+1]);
				fill(Xv[l1+1],Xv[l1+1] + (Av[l1+1]->nrow),0.0);
				l1++;
			}

			//Direct Solve
			Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
			l1 = l;

			while(l1 > 0)
			{
				parallel::transfer_solution(*Pv[l1-1],Xv[l1],Xv[l1-1]);
				parallel::jacobi_smoother(*Av[l1-1],Bv[l1-1],Xv[l1-1],smooth_iter);
				l1 = l1 - 1;
			}

			cycles++;
			r1 = parallel::residual(*Av[l1],Bv[l1],Xv[l1]);

            if(print_solve_phase_details == 1)
            {
                cout<<cycles<<" "<<r1<<endl;
            }	
		}
    }

	copy(Xv[0],Xv[0]+Av[0]->nrow,x);

};


//Method to solve phase of AMG with SOR as smoother
void AMG_solver::AMG_solve_SOR(double* &b,double* &x,int iterations)
{
	int cycles = 0;
	double r1  = 0.0;

    copy(b,b+Av[0]->nrow,Bv[0]);
	copy(x,x+Av[0]->nrow,Xv[0]);
	
	r1 = parallel::residual(*Av[0],Bv[0],Xv[0]);
    int l1 = 0;

    if(iterations > 0)
    {
      	while(cycles++ < iterations)
		{
	 		while(l1 < l)
     		{
	 			parallel::sor_smoother(*Av[l1],Bv[l1],Xv[l1],6);
	 			parallel::store_residual(*Av[l1],Bv[l1],Xv[l1],Rv[l1]);
	 			parallel::transfer_residual(*Pv[l1],Rv[l1],Bv[l1+1]);
	 			fill(Xv[l1+1],Xv[l1+1] + (Av[l1+1]->nrow),0.0);
	 			l1++;
	 		}

			//Direct Solve
			Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
			l1 = l;

			while(l1 > 0)
			{
				parallel::transfer_solution(*Pv[l1-1],Xv[l1],Xv[l1-1]);
				parallel::sor_smoother(*Av[l1-1],Bv[l1-1],Xv[l1-1],6);
				l1 = l1 - 1;
			}

			cycles++;
			r1 = parallel::residual(*Av[l1],Bv[l1],Xv[l1]);
			
		}
    }
	
	if(iterations == -1)
    {
        while(r1 > tol1)
		{
			while(l1 < l)
			{
				parallel::sor_smoother(*Av[l1],Bv[l1],Xv[l1],6);
				parallel::store_residual(*Av[l1],Bv[l1],Xv[l1],Rv[l1]);
				parallel::transfer_residual(*Pv[l1],Rv[l1],Bv[l1+1]);
				fill(Xv[l1+1],Xv[l1+1] + (Av[l1+1]->nrow),0.0);
				l1++;
			}

			//Direct Solve
			Directsolve->Direct_Solver_Pardiso_solve(Bv[l],Xv[l]);
			l1 = l;

			while(l1 > 0)
			{
				parallel::transfer_solution(*Pv[l1-1],Xv[l1],Xv[l1-1]);
				parallel::sor_smoother(*Av[l1-1],Bv[l1-1],Xv[l1-1],6);
				l1 = l1 - 1;
			}

			cycles++;
			r1 = parallel::residual(*Av[l1],Bv[l1],Xv[l1]);
			cout<<cycles<<"\t"<<r1<<endl;
			
		}
    }

};
