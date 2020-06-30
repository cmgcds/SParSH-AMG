
#ifndef AMG_MATRIX_HPP_
#define AMG_MATRIX_HPP_

//Base Class for storing Matrices in CSR Form
class sp_matrix
{
    public:

        int nrow;		//Number of Rows
        int ncol;	    //Number of Columns
        int nnz;		//Number of Non-zeros

        int *rowptr   = nullptr;	//Row Pointer
        int *colindex = nullptr;    //Column Index
        double *val   = nullptr;    //Value Array

    public:

    //Constructor to allocate memory for Matrix in CSR Form
    sp_matrix(int r,int c,int n);

    sp_matrix();

    //Prints the Matrix Stored in CSR Form
    void check_sp_matrix();

    //Frees the memory allocated to 3 arrays in CSR Form
    //~sp_matrix();


};

#endif /* AMG_HPP_ */
