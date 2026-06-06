#ifndef BITMATRIX_HPP
#define BITMATRIX_HPP

#include <vector>
#include <random>
#include <string>

class bitmatrix;
class bitvector;

bitvector operator* (const bitmatrix &M, const bitvector &V); //matrix times vector mod 2
bitvector operator* (const bitvector &V, const bitvector &M); //vector times matrix mod 2
bitvector operator* (const bitvector &V, const bool &s); //vector times scalar
bitvector operator* (const bool &s, const bitvector &V); //scalar times vector
bitmatrix operator* (const bitmatrix &M, const bool &s); //matrix times scalar
bitmatrix operator* (const bool &s, const bitmatrix &M); //scalar times matrix

class bitmatrix {
    public:
        bitmatrix(); //constructor for empty bitmatrix
        bitmatrix(size_t rows, size_t cols); //constructor for bitmatrix of size rows x cols containing all zeros
        ~bitmatrix(); //destructor
        size_t num_rows() const; //returns number of rows in bitmatrix
        size_t num_cols() const; //returns number of columns in bitmatrix
        size_t count_ones() const; //returns the number of ones in the bitmatrix
        void resize(size_t rows, size_t cols); //resizes the bitmatrix and overwrites with all zeros
        void from_sparse_rows(const std::vector <std::vector <size_t> > &ones, size_t num_rows, size_t num_cols); //ones[i][j] is the location of the jth one in row i
        void from_sparse_cols(const std::vector <std::vector <size_t> > &ones, size_t num_rows, size_t num_cols); //ones[i][j] is the location of the jth one in col i
        std::vector <std::vector <size_t> > to_sparse_rows(); //returnval[i][j] is the location of the jth one in row i
        std::vector <std::vector <size_t> > to_sparse_cols(); //returnval[i][j] is the location of the jth one in col i
        void row_add(size_t i, size_t j);  //elementary row operation: add row i to row j
        void col_add(size_t i, size_t j);  //elementary col operation: add col i to col j
        void row_swap(size_t i, size_t j); //elementary row operation: swap row i with row j
        void col_swap(size_t i, size_t j); //elementary col operation: swap col i with col j
        void row_zero(size_t i); //elementary row operation: multiply row i by zero
        void col_zero(size_t i); //elementary col operation: mutiply col i by zero
        void identity(size_t n); //make the n x n identity matrix, resizing if necessary
        void zeros(size_t rows, size_t cols); //make the rows x cols zero matrix, resizing if necessary
        void random(std::mt19937_64 &eng); //fill in the existing matrix with random entries
        void random(size_t rows, size_t cols, std::mt19937_64 &eng); //make a random rows x cols random matrix, resizing if necessary
        std::string to_str() const; //output the matrix in the form of a string
        bool from_str(std::string s, char separator); //load the matrix from a string
        bool load_dense(std::string filename); //load bitmatrix from ascii file in dense format
        void save_dense(std::string filename); //save bitmatrix to ascii file in dense format
        bool load_sparse(std::string filename); //load bitmatrix from ascii file in sparse format
        void save_sparse(std::string filename); //save bitmatrix to ascii file in sparse format
        bool get(size_t i, size_t j) const; //get entry i,j
        void set(size_t i, size_t j, bool val); //set entry i,j to val
        void flip(size_t i, size_t j); //flip entry i,j
        bitmatrix transpose() const; //output the transpose of the matrix
        void operator= (const bitmatrix &M); //overloaded assignment operator
        bool operator== (const bitmatrix &M) const; //overloaded equality operator
        bool operator!= (const bitmatrix &M) const; //overloaded inequality operator
        friend bitmatrix operator+ (const bitmatrix &M1, const bitmatrix &M2); //add matrices mod 2
        bitmatrix &operator+= (const bitmatrix &M); //in place addition mod 2
        friend bitmatrix operator* (const bitmatrix &M1, const bitmatrix &M2); //matrix multiplication mod 2
        friend bitvector operator* (const bitmatrix &M, const bitvector &V); //matrix times vector mod 2
        friend bitvector operator* (const bitvector &V, const bitmatrix &M); //vector times matrix mod 2
        friend bitmatrix operator& (const bitmatrix &M1, const bitmatrix &M2); //entrywise AND
        friend bitmatrix operator| (const bitmatrix &M1, const bitmatrix &M2); //entrywise OR
        bitmatrix operator~ () const; //entrywise NOT
        friend std::ostream& operator<<(std::ostream& os, const bitmatrix& M); //stream bitmatrix.to_str() e.g. to cout
        friend size_t Z_dot(const bitmatrix &M1, const bitmatrix &M2); //dot product over the integers
        friend bool Z2_dot(const bitmatrix &M1, const bitmatrix &M2); //dot product mod 2
        void append_right(const bitmatrix &M); //append M to the right of the bitmatrix
        void append_below(const bitmatrix &M); //append M below the bitmatrix
        void append_right(const bitvector &V); //append V to the right of the bitmatrix
        void append_below(const bitvector &V); //append V below the bitmatrix
        friend bitmatrix vstack(const bitmatrix &M1, const bitmatrix &M2); //append M2 below M1
        friend bitmatrix hstack(const bitmatrix &M1, const bitmatrix &M2); //append M2 to the right of M1
        bitmatrix col_range(size_t start, size_t end) const; //output the submatrix with columns start <= j < end
        bitmatrix row_range(size_t start, size_t end) const; //output the submatrix with rows start <= i < end
        bitmatrix row(size_t i) const; //return the ith row as a 1 by n_cols matrix
        bitmatrix col(size_t j) const; //return the jth column as a n_rows by 1 matrix
        bitvector row_vec(size_t i) const; //return the ith row as a bitvector
        bitvector col_vec(size_t i) const; //return the ith column as a bitvector
        bitmatrix col_transpose(size_t j) const; //return the jth column as an 1 by n_cols matrix (more efficient to work with since bitmatrix is row-major)
        bitmatrix inverse(bool &invertible) const; //return the matrix inverse if it exists, write false into invertible if it does not
        bitmatrix row_echelon_form() const; //row echelon form (does not wipe out the nonzeros above the pivots)
        bitmatrix reduced_row_echelon_form() const; //reduced row echelon form (wipes out the nonzeros above the pivots)
        void row_echelon_decomp(bitmatrix &G, bitmatrix &R) const; //GM = R. M is this bitmatrix. R is in row echelon form.
        void reduced_row_echelon_decomp(bitmatrix &G, bitmatrix &R) const; //GM = R. M is this bitmatrix. R is in reduced row echelon form.
        int rank() const; //the matrix rank
        bool determinant() const; //returns the determinant if the matrix is square
        friend class bitvector;
        friend size_t new_weight(const bitmatrix &B, const bitvector &v, size_t i); //the new Hamming weight of v if we added the ith row of B to it
    private:
        std::vector <uint64_t> blocks;
        size_t n_cols;
        size_t n_rows;
        size_t n_blocks;
        size_t blocks_per_row;
};

class bitvector {
    public:
        bitvector(); //constructor for empty bitvector
        bitvector(size_t length); //constructor for bitvector of specified length containing all zeros
        ~bitvector(); //destructor
        size_t size() const; //returns number entries in bitvector
        size_t count() const; //returns the number of ones in the bitvector
        void resize(size_t length); //resizes the bitvector and overwrites with all zeros
        void from_sparse(const std::vector <size_t> &ones, size_t length); //ones[i] is the location of the ith one
        std::vector <size_t> to_sparse(); //returnval[i] is the location of the ith one
        void zeros(size_t length); //make the all-zeros vector of specified length, resizing if necessary
        void random(std::mt19937_64 &eng); //fill in the existing vector with random entries
        void random(size_t length, std::mt19937_64 &eng); //make a random vector of given length, resizing if necessary
        std::string to_str() const; //output the vector in the form of a string (little endian)
        bool from_str(std::string s); //load the vector from a string of 1s and 0s (little endian)
        bool load_dense(std::string filename); //load bitvector from ascii file in dense format (little endian)
        void save_dense(std::string filename); //save bitvector to ascii file in dense format (little endian)
        bool load_sparse(std::string filename); //load bitvector from ascii file in sparse format
        void save_sparse(std::string filename); //save bitvector to ascii file in sparse format
        bool get(size_t i) const; //get entry i
        void set(size_t i, bool val); //set entry i to val
        void flip(size_t i); //flip entry i
        void operator= (const bitvector &V); //overloaded assignment operator
        bool operator== (const bitvector &V) const; //overloaded equality operator
        bool operator!= (const bitvector &V) const; //overloaded inequality operator
        friend bitvector operator+ (const bitvector &V1, const bitvector &V2); //add vectors mod 2
        bitvector &operator+= (const bitvector &V); //in place addition mod 2
        friend bitvector operator& (const bitvector &V1, const bitvector &V2); //entrywise AND
        friend bitvector operator| (const bitvector &V1, const bitvector &V2); //entrywise OR
        bitvector operator~ () const; //entrywise NOT
        void increment(); //increment as if it were a number
        void from_num(uint64_t num, size_t length); //make a bitvector from a number using little-endian convention
        uint64_t to_num(); //convert bitvector to a number if length is at most 64. If length > 64 generate error message.
        friend std::ostream& operator<<(std::ostream& os, const bitvector& V); //stream bitvector.to_str() e.g. to cout
        friend size_t Z_dot(const bitvector &V1, const bitvector &V2); //dot product over the integers
        friend bool Z2_dot(const bitvector &M1, const bitvector &V2); //dot product mod 2
        friend bitvector operator* (const bitmatrix &M, const bitvector &V); //matrix times vector mod 2
        friend bitvector operator* (const bitvector &V, const bitmatrix &M); //vector times matrix mod 2
        void append(const bitvector &V); //append V to the right of the bitvector
        void push_back(bool val); //add one more entry with value val to the end of the vector
        friend bitvector hstack(const bitvector &V1, const bitvector &V2); //append V2 to the right of V1
        bitvector range(size_t start, size_t end) const; //output the subvector with entries in the range: start <= j < end
        bitmatrix to_row_matrix(); //convert to 1 by n bitmatrix
        bitmatrix to_col_matrix(); //convert to n by 1 bitmatrix
        bitvector complement() const; //returns the bitwise NOT 
        friend class bitmatrix;
        friend size_t new_weight(const bitmatrix &B, const bitvector &v, size_t i); //the new Hamming weight of v if we added the ith row of B to it
    private:
        std::vector <uint64_t> blocks;
        size_t n_bits;
        size_t n_blocks;
};

//reduces M in-place to reduced row echelon form, using only the columns 0..cmax-1
//returns rank
size_t rref(bitmatrix &M, size_t cmax);

//reduces M in-place to row echelon form (not reduced), using only columns 0..cmax-1
//returns rank
size_t ref(bitmatrix &M, size_t cmax);

bitmatrix permute_rows(const bitmatrix &M, const std::vector<size_t> &permutation);

bitmatrix permute_columns(const bitmatrix &M, const std::vector<size_t> &permutation);

bitvector permute_bits(const bitvector &v, const std::vector<size_t> &permutation);

bitvector rand_weight_k(std::mt19937_64 &eng, size_t blocksize, size_t error_weight);

bitmatrix submatrix(const bitmatrix &B, const bitvector &sub);

bitmatrix direct_sum(const bitmatrix &A, const bitmatrix &B);

bitmatrix direct_sum(const std::vector<bitmatrix> &blocks);

bitmatrix block(const bitmatrix &B, size_t row_start, size_t row_end, size_t col_start, size_t col_end);

#endif
