# SParSH: Scalable PARallel numerics for Heterogeneous systems 

#### A Library for Hybrid CPU-GPU Algebraic Multigrid and Preconditioned Iterative Methods

<p align="justify"> SParSH is hybrid CPU-GPU based solver library featuring parallel Algebraic Multigrid (AMG) solvers and preconditioners for solving large sparse linear system of equations. The library provides implementations of AMG as solver and AMG-preconditioned Krylov subspace solvers for multicore CPU and single node CPU-GPU based computing environments. These implementations utilize hybrid CPU-GPU based approaches to accelerate solving of sparse linear systems. The SParSH package offers flexibility to construct required solver and preconditioner which can be integrated into existing scientific applications.       


Key features of library are: 

- Classical (Becks coarsening) , Unsmoothened aggregation and smoothened aggregation based algebraic multigrid
- OpenMP parallelized implementation of AMG solvers and preconditioners
- Hybrid CPU-GPU based implementation of AMG solvers and preconditioners
- Krylov Solvers: CG, GMRES and BiCGStab
- Smoothers: Jacobi, SOR (currently limited)



## Table of Contents

- [Dependencies and Requirements](#depend)
- [Building Package and Quick start guide](#build)
- [Further Reading and Documentation](#further)

## <a name="depend"></a>Dependencies and Requirements 
In order to build library, user would need the CMake, CUDA Toolkit and Intel MKL
Recommended Versions: CMake >= 3.9 , CUDA Toolkit 10.0, Intel® Math Kernel Library 2020 

## <a name="build"></a>Building Package and Quick start guide 
Following steps lists the procedure by which one can use library in existing cmake project

1. Specify the following in CMakeLists.txt 

   - Specify C Compiler path in CMAKE_C_COMPILER variable and C++ Compiler path in CMAKE_CXX_COMPILER variable 
     Ex. 

     ```cmake
     set(CMAKE_C_COMPILER   "/opt/intel/compilers_and_libraries/linux/bin/intel64/icc")
     set(CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries/linux/bin/intel64/icpc")
     ```

   - Specify path for include directories for mkl and cuda
     Ex. 

     ```cmake
     include_directories("/opt/intel/mkl/include")
     include_directories("/usr/local/cuda/include")
     ```

   - Specify path for link directories for mkl and cuda
     Ex. 

     ```cmake
     link_directories("/opt/intel/mkl/lib/intel64")
     link_directories("/opt/intel/lib/intel64")
     link_directories("/usr/local/cuda-10.0/lib64")
     link_directories("/usr/local/lib")
     ```

2. Copy include and src directories to cmake project and add the following in CMakeLists.txt

   Ex.

   ```cmake
   include_directories("${CMAKE_SOURCE_DIR}/include")
   add_subdirectory(src)
   ```

3. Given that main.cpp is file of your cmake project in which solver needs to be called, do the following

   - include header "AMG.hpp" in the cpp file

   - Given that you have access to matrix stored in Compressed Sparse Row (CSR) format, create an object of sp_matrix_mg datatype to hold matrix and then call solver routines

     Ex.

     ```c++
     sp_matrix_mg *A = new sp_matrix_mg();
     A->nrow = nrows;      		//Specify number of rows of matrix
     A->ncol = ncols;	  		//Specify number of columns of matrix
     A->nnz  = nnz;		  		//Specify number of non zeros
     
     A->rowptr   = rowptr   		//Specify rowptr array of matrix 0 based indexing
     A->colindex = columnindex	//Specify column index array of matrix
     A->val      = val			//Specify value array of matrix
         
     A->sp_matrix_fill();		//Helper functions to complete matrix object creation
     A->sp_matrix_fill_diagonal()//Helper functions to complete matrix object creation
         
         
     //Call solver routine
     //b: double array which holds rhs of Ax=b linear system
     //x: solution array
     AMG_Solver_1(*A,b,x);
     ```       

4. Specify the following linker flags along with your existing linker flags in CMakeLists.txt
   Also specify the link libraries in the following order with your existing libraries in CMakeLists.txt

   Ex. if main is target executable of the cmake project, 

   ```cmake
   set(CMAKE_EXE_LINKER_FLAGS "-ldl -lpthread -lcusparse -lculibos -lcublas ")     
   target_link_libraries(main
       PUBLIC amg m mkl_intel_lp64 mkl_intel_thread mkl_core irc svml iomp5)
   ```

CMakeList.txt in the main repository builds an example project from main.cpp which reads matrix from input file which is in matrix market format and calls the solver routine.  

## <a name="further"></a> Further reading and Documentation

Header file AMG.hpp provides access to the solver parameters which can be modified as per requirements. It includes declaration of available solver routines and their description. 
