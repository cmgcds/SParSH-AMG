
#include "AMG_coarsening.hpp"
#include "AMG_smoothers.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

namespace sequential
{
    //Heavy Edge Matching based Coarsening
	void HEM_Prolongator(sp_matrix_mg& A,sp_matrix_mg*  &P,int l1)
	{
		P = new sp_matrix_mg(A.nrow,1,A.nrow);
		int newnum = 0;

		//Initialization of Variables
		std::fill(P->colindex,P->colindex+A.nrow,-1);
		std::fill(P->val,P->val+A.nrow,1.0);
		double max1 = 0;
		int id      = 0;

		if(l1%2 == 0)
		{
			for(int i = 0;i < A.nrow;i++)
			{
				P->rowptr[i] = i;
				if(P->colindex[i]==-1)
				{
					id = -1;
					max1 = 0.0;

					//Finds the Heaviest Neighbor
					for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
					{
						if(P->colindex[A.colindex[j]] == -1 && fabs(A.val[j]) > max1 && A.colindex[j] !=i )
						{
							max1 = fabs(A.val[j]);
							id   = A.colindex[j];
						}
					}

					if(id != -1)
					{
						P->colindex[i]  = newnum;
						P->colindex[id] = newnum;
						newnum++;
					}	
				}
			}
		}
		else
		{
		 	for(int i = A.nrow-1;i >= 0;i--)
			{
				P->rowptr[i] = i;
				if(P->colindex[i]==-1)
				{
					id = -1;
					max1 = 0.0;

					//Finds the Heaviest Neighbor
					for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
					{
						if(P->colindex[A.colindex[j]] == -1 && fabs(A.val[j]) > max1 && A.colindex[j] !=i )
						{
							max1 = fabs(A.val[j]);
							id   = A.colindex[j];
						}
					}

					if(id != -1)
					{
						P->colindex[i]  = newnum;
						P->colindex[id] = newnum;
						newnum++;
					}	
				}
			} 
		}
		
		for(int i=0; i < P->nrow; i++)
		{
			if(P->colindex[i]==-1)
			{
				P->colindex[i]= newnum;
				newnum++;
			}
		}

		P->rowptr[P->nrow] = P->nrow;
		P->ncol = newnum;

		P->sp_matrix_fill();
	}
    
    //Helper Functions for MIS Based coarsening
    struct match
	{
		int s;
		int v;
		int index;
	};

	void max_match(match &a,match &b)
	{
		if(a.s < b.s)
		{
			a.s = b.s;
			a.v = b.v;
			a.index = b.index;
		}
		else if(a.s == b.s)
		{
			if(a.v < b.v)
			{
				a.s = b.s;
				a.v = b.v;
				a.index = b.index;
			}
			else if(a.v == b.v)
			{
				if(a.index < b.index)
				{
					a.s = b.s;
					a.v = b.v;
					a.index = b.index;
				}
			}
		}
	}
     
     //MIS based coarsening
    void mis_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P1)
	{

		//Initialization
		P1 = new sp_matrix_mg(A.nrow,1,A.nrow);
		for(int i = 1; i < A.nrow + 1;i++)
		{
			P1->rowptr[i] = i;
			P1->val[i-1] = 1.0;
			P1->colindex[i-1] = 0;

		}

		match* mp = new match[A.nrow];

		for(int i = 0; i < A.nrow; i++)
		{
			mp[i].s = P1->colindex[i];
			mp[i].v = rand();
			mp[i].index = i;

		}

		//Main Loop
		int count = 1;
		match* mp_update = new match[A.nrow];
		do
		{
			count = 1;
			for(int i = 0; i < A.nrow;i++)
			{
				if(P1->colindex[i] == 0)
				{
					count = 0;
					break;
				}
			}


			for(int i = 0; i < A.nrow; i++)
			{
				mp[i].s = P1->colindex[i];
				mp_update[i].s = mp[i].s;
				mp_update[i].v = mp[i].v;
				mp_update[i].index = mp[i].index;
			}


			int distance = 2;
			while(distance > 0)
			{

				match* mp_update1 = new match[A.nrow];
				match sp;
				for(int i = 0;i < A.nrow;i++)
				{
					if(mp[i].s == 0)
					{
						sp.s = mp_update[i].s;
						sp.v = mp_update[i].v;
						sp.index = mp_update[i].index;

						for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
						{
							max_match(sp,mp_update[A.colindex[j]]);
						}

						mp_update1[i].s = sp.s;
						mp_update1[i].v = sp.v;
						mp_update1[i].index = sp.index;
					}
				}

				for(int i = 0; i < A.nrow; i++)
				{
					mp_update[i].s = mp_update1[i].s;
					mp_update[i].v = mp_update1[i].v;
					mp_update[i].index = mp_update1[i].index;

				}
				delete[] mp_update1;
				distance = distance - 1;
			}

			for(int i = 0; i < A.nrow; i++)
			{
				if(P1->colindex[i] == 0)
				{
					if(mp_update[i].index==i)
					{
						P1->colindex[i] = 1;
					}
					else if(mp_update[i].s == 1)
					{
						P1->colindex[i] = -1;
					}
				}
			}
		}while(count == 0);

		delete[] mp_update;
		delete[] mp;

		for(int i = 0; i < A.nrow; i++)
		{
			if(P1->colindex[i] < 0)
			{
				P1->colindex[i] = 0;
			}

		}

		int c = 0;
		for(int i = 0; i < A.nrow; i++)
		{
			if(P1->colindex[i]==1)
			{
				for(int j = A.rowptr[i]; j < A.rowptr[i+1];j++)
				{
					P1->colindex[A.colindex[j]] = c;
				}
				c = c + 1;
			}

		}

		P1->ncol = c;
		P1->sp_matrix_fill();
		
		
	}

    //Becks coarsening Algorithm
	void beck_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P1)
	{
		vector<int> c_f(A.nrow);
		fill(c_f.begin(),c_f.end(),0);
		int c_count = 0;
		for(int i = 0;i < A.nrow;i++)
		{
			if(c_f[i] == 0)
			{
				for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
				{
					c_f[A.colindex[j]] -= 1;
				}
				c_f[i] = c_count+1;
				c_count++;
			}
		}

		vector<int> rowp(A.nrow+1,0);
		vector<int> colp;
		vector<double> val;

		for(int i = 0;i < A.nrow;i++)
		{
			if(c_f[i] > 0)
			{
				rowp[i+1] = 1;
				colp.push_back(c_f[i]-1);
				val.push_back(1.0);
				continue;
			}
			
			if(c_f[i] < 0)
			{
				for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
				{
					int k = A.colindex[j];
					double p1 = 1/fabs(c_f[i]);
					if(c_f[k] > 0)
					{
						rowp[i+1]++;
						colp.push_back(c_f[k]-1);
						val.push_back(p1);
					}

				}
			}	
		}

		for(int i = 0;i < A.nrow;i++)
		{
			rowp[i+1] += rowp[i];
		}

		P1  = new sp_matrix_mg(A.nrow,c_count,val.size());
		std::copy(rowp.begin(),rowp.end(),P1->rowptr);
		std::copy(colp.begin(),colp.end(),P1->colindex);
		std::copy(val.begin() ,val.end(),P1->val);

		P1->sp_matrix_fill();
		// cout<<P1->nrow<<endl;
		// cout<<P1->ncol<<endl;
		// cout<<P1->nnz<<endl;
		//A.check_sp_matrix();
		//P->check_sp_matrix();

		rowp.clear();
		colp.clear();
		val.clear();
		c_f.clear();
	}

	//Compatible weighted matching based coarsening
    void C_W_prolongator(sp_matrix_mg &A,sp_matrix_mg *&P,int l1)
	{
		P = new sp_matrix_mg(A.nrow,1,A.nrow);
		int newnum = 0;
		std::fill(P->val,P->val+A.nrow,1.0);
		double* rhs = new double[A.nrow];
		std::fill(rhs,rhs+A.nrow,0.0);
		parallel::jacobi_smoother(A,rhs,P->val,10);

		//Initialization of Variables
		std::fill(P->colindex,P->colindex+A.nrow,-1);
		
		double max1 = 0;
		int id      = 0;
		double norm = 0;

		if(l1%2 == 0)
		{
			for(int i = 0;i < A.nrow;i++)
			{
				P->rowptr[i] = i;
				if(P->colindex[i]==-1)
				{
					id = -1;
					max1 = 0.0;

					//Finds the Heaviest Neighbor
					for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
					{
						if(P->colindex[A.colindex[j]] == -1 && fabs(A.val[j]) > max1 && A.colindex[j] !=i )
						{
							max1 = fabs(A.val[j]);
							id   = A.colindex[j];
						}
					}

					if(id != -1)
					{
						P->colindex[i]  = newnum;
						P->colindex[id] = newnum;
						norm = sqrt(P->val[i]*P->val[i] + P->val[id]*P->val[id]);
						P->val[i]  /= norm;
						P->val[id] /= norm;
						newnum++;
					}	
				}
			}

			for(int i=0; i < P->nrow; i++)
			{
				if(P->colindex[i]==-1)
				{
					P->colindex[i] = newnum;
					P->val[i]      /= fabs(P->val[i]);
					newnum++;
				}
			}
			P->rowptr[P->nrow] = P->nrow;
			P->ncol = newnum;
		}
		else
		{
			for(int i = A.nrow-1;i >= 0;i--)
			{
				P->rowptr[i] = i;
				if(P->colindex[i]==-1)
				{
					id = -1;
					max1 = 0.0;

					//Finds the Heaviest Neighbor
					for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
					{
						if(P->colindex[A.colindex[j]] == -1 && fabs(A.val[j]) > max1 && A.colindex[j] !=i )
						{
							max1 = fabs(A.val[j]);
							id   = A.colindex[j];
						}
					}

					if(id != -1)
					{
						P->colindex[i]  = newnum;
						P->colindex[id] = newnum;
						norm = sqrt(P->val[i]*P->val[i] + P->val[id]*P->val[id]);
						P->val[i]  /= norm;
						P->val[id] /= norm;
						newnum++;
					}	
				}
			}

			for(int i=0; i < P->nrow; i++)
			{
				if(P->colindex[i]==-1)
				{
					P->colindex[i] = newnum;
					P->val[i]      /= fabs(P->val[i]);
					newnum++;
				}
			}
			P->rowptr[P->nrow] = P->nrow;
			P->ncol = newnum;
			
		}
		
		P->sp_matrix_fill();
		delete[] rhs;
	}

	struct edge_list
	{
		int i;
		int j;
		double s;
	};

	int compare_edge_list(const void* a1,const void* b1)
	{
		edge_list* a = (edge_list*)a1;
		edge_list* b = (edge_list*)b1;

		if(a->s != b->s)
		{
			return (a->s < b->s);
		}
		else
		{
			return (a->i > b->i);
		}
		return 0;
	}

	//Coarsening based on HEM 2 Heuristic prolongation
	void HEM_Prolongator_2(sp_matrix_mg& A,sp_matrix_mg*  &P)
	{
		P = new sp_matrix_mg(A.nrow,1,A.nrow);
		int newnum = 0;

		//Initialization of Variables
		std::fill(P->colindex,P->colindex+A.nrow,-1);
		std::fill(P->val,P->val+A.nrow,1.0);
		double max1 = 0;
		int id      = 0;

		edge_list* matrix_entries = new edge_list[A.nnz - A.nrow];
		int index = 0;

		for(int i = 0;i < A.nrow;i++)
		{
			P->rowptr[i] = i;
			for(int j = A.rowptr[i];j < A.rowptr[i+1];j++)
			{
				if(i != A.colindex[j])
				{
					matrix_entries[index].i = i;
					matrix_entries[index].j = A.colindex[j];
					matrix_entries[index].s = fabs(A.val[j]);
					index++;	
				}
			}
		}

		//Sort the edge list array
		qsort(matrix_entries,A.nnz-A.nrow,sizeof(edge_list),compare_edge_list);

		int coarse_dof_number = 0;
		int v;
		int w;

		//Populate P matrix based on sorted entries of edge lists
		for(int k = 0;k < A.nnz-A.nrow;k++)
		{
			v = matrix_entries[k].i;
			w = matrix_entries[k].j;

			if(P->colindex[v] == -1 && P->colindex[w] == -1)
			{
				P->colindex[v] = coarse_dof_number;
				P->colindex[w] = coarse_dof_number;
				coarse_dof_number++;
			}
		}
		
		for(int i=0; i < P->nrow; i++)
		{
			if(P->colindex[i]==-1)
			{
				P->colindex[i]= coarse_dof_number;
				coarse_dof_number++;
			}
		}

		P->rowptr[P->nrow] = P->nrow;
		P->ncol = coarse_dof_number;

		P->sp_matrix_fill();

	}

}