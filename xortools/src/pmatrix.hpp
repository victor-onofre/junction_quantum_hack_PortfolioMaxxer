#ifndef PMATRIX_HPP
#define PMATRIX_HPP

#include <vector>
#include <random>
#include <string>

class pmatrix;
class pvector;

pvector operator* (const pmatrix &M, const pvector &V); //matrix times vector mod p
pvector operator* (const pvector &V, const pvector &M); //vector times matrix mod p
pvector operator* (const pvector &V, const uint16_t &s); //vector times scalar
pvector operator* (const uint16_t &s, const pvector &V); //scalar times vector
pmatrix operator* (const pmatrix &M, const uint16_t &s); //matrix times scalar
pmatrix operator* (const uint16_t &s, const pmatrix &M); //scalar times matrix

typedef struct {
    uint16_t val;
    size_t loc;
}entry;

class pmatrix {
    public:
        pmatrix(); //constructor for empty bitmatrix with unknown modulus
        pmatrix(uint16_t p); //constructor for empty bitmatrix
        pmatrix(size_t rows, size_t cols, uint16_t p); //constructor for bitmatrix of size rows x cols containing all zeros
        ~pmatrix(); //destructor
        void set_modulus(uint16_t p); //set the modulus to p, which must be a prime smaller than PMAX
        uint16_t p() const; //returns the modulus
        size_t num_rows() const; //returns number of rows in bitmatrix
        size_t num_cols() const; //returns number of columns in bitmatrix
        size_t nonzeros() const; //the number of nonzero entries
        void resize(size_t rows, size_t cols); //resizes the bitmatrix and overwrites with all zeros
        void from_sparse_rows(const std::vector <std::vector <entry> > &nonzeros, size_t num_rows, size_t num_cols); //nonzeros[i][j] is the jth nonzero entry in row i
        void from_sparse_cols(const std::vector <std::vector <entry> > &nonzeros, size_t num_rows, size_t num_cols); //nonzeros[i][j] is the jth nonzero entry in col i
        std::vector <std::vector <entry> > to_sparse_rows(); //returnval[i][j] is the jth nonzero entry in row i
        std::vector <std::vector <entry> > to_sparse_cols(); //returnval[i][j] is the jth nonzero entry in col i
        void row_add(size_t i, size_t j, uint16_t m);  //elementary row operation: add m times row i to row j
        void col_add(size_t i, size_t j, uint16_t m);  //elementary col operation: add m times col i to col j
        void row_swap(size_t i, size_t j); //elementary row operation: swap row i with row j
        void col_swap(size_t i, size_t j); //elementary col operation: swap col i with col j
        void row_mult(size_t i, uint16_t m); //elementary row operation: multiply row i by m
        void col_mult(size_t i, uint16_t m); //elementary col operation: mutiply col i by m
        void identity(size_t n); //make the n x n identity matrix, resizing if necessary
        void zeros(size_t rows, size_t cols); //make the rows x cols zero matrix, resizing if necessary
        void random(std::mt19937_64 &eng); //fill in the existing matrix with random entries
        void random(size_t rows, size_t cols, std::mt19937_64 &eng); //make a rows x cols random matrix, resizing if necessary
        std::string to_str() const; //output the matrix in the form of a string
        bool from_str(std::string s, char separator); //load the matrix from a string
        bool load_dense(std::string filename); //load bitmatrix from ascii file in dense format
        void save_dense(std::string filename); //save bitmatrix to ascii file in dense format
        bool load_sparse(std::string filename); //load bitmatrix from ascii file in sparse format
        void save_sparse(std::string filename); //save bitmatrix to ascii file in sparse format
        uint16_t get(size_t i, size_t j) const; //get entry i,j
        void set(size_t i, size_t j, uint16_t val); //set entry i,j to val
        pmatrix transpose() const; //output the transpose of the matrix
        void operator= (const pmatrix &M); //overloaded assignment operator
        bool operator== (const pmatrix &M) const; //overloaded equality operator
        bool operator!= (const pmatrix &M) const; //overloaded inequality operator
        friend pmatrix operator+ (const pmatrix &M1, const pmatrix &M2); //add matrices mod p
        friend pmatrix operator- (const pmatrix &M1, const pmatrix &M2); //subtract matrices mod p
        pmatrix &operator+= (const pmatrix &M); //in place addition mod p
        pmatrix &operator-= (const pmatrix &M); //in place addition mod p
        friend pmatrix operator* (const pmatrix &M1, const pmatrix &M2); //matrix multiplication mod p
        friend pvector operator* (const pmatrix &M, const pvector &V); //matrix times vector mod p
        friend pvector operator* (const pvector &V, const pmatrix &M); //vector times matrix mod p
        friend std::ostream& operator<<(std::ostream& os, const pmatrix& M); //stream bitmatrix.to_str() e.g. to cout
        friend uint16_t Zp_dot(const pmatrix &M1, const pmatrix &M2); //dot product mod p
        void append_right(const pmatrix &M); //append M to the right of the bitmatrix
        void append_below(const pmatrix &M); //append M below the bitmatrix
        void append_right(const pvector &V); //append V to the right of the bitmatrix
        void append_below(const pvector &V); //append V below the bitmatrix
        friend pmatrix vstack(const pmatrix &M1, const pmatrix &M2); //append M2 below M1
        friend pmatrix hstack(const pmatrix &M1, const pmatrix &M2); //append M2 to the right of M1
        pmatrix col_range(size_t start, size_t end) const; //output the submatrix with columns start <= j < end
        pmatrix row_range(size_t start, size_t end) const; //output the submatrix with rows start <= i < end
        pmatrix row(size_t i) const; //return the ith row as a 1 by n_cols pmatrix
        pmatrix col(size_t j) const; //return the jth column as a n_rows by 1 pmatrix
        pvector row_vec(size_t i) const; //return the ith row as a pvector
        pvector col_vec(size_t i) const; //return the ith column as a pvector
        pmatrix col_transpose(size_t j) const; //return the jth column as an 1 by n_cols matrix (more efficient to work with since bitmatrix is row-major)
        pmatrix inverse(bool &invertible) const; //return the matrix inverse if it exists, write false into invertible if it does not
        pmatrix row_echelon_form() const; //row echelon form (does not wipe out the nonzeros above the pivots)
        pmatrix reduced_row_echelon_form() const; //reduced row echelon form (wipes out the nonzeros above the pivots)
        void row_echelon_decomp(pmatrix &G, pmatrix &R) const; //GM = R. M is this bitmatrix. R is in row echelon form.
        void reduced_row_echelon_decomp(pmatrix &G, pmatrix &R) const; //GM = R. M is this bitmatrix. R is in reduced row echelon form.
        int rank() const; //the matrix rank
        uint16_t determinant() const; //returns the determinant if the matrix is square
        friend class pvector;
        void clear();
    private:
        uint16_t modulus;
        std::vector <uint16_t> entries;
        size_t n_cols;
        size_t n_rows;
};

class pvector {
    public:
        pvector(); //constructor for empty bitvector with unknown modulus
        pvector(uint16_t p); //constructor for empty bitvector
        pvector(size_t length, uint16_t p); //constructor for pvector of specified length containing all zeros
        ~pvector(); //destructor
        void set_modulus(uint16_t p); //set the modulus to p, which must be a prime smaller than PMAX
        uint16_t p() const; //returns the modulus
        size_t size() const; //returns number entries in pvector
        size_t nonzeros() const; //the number of nonzero entries
        size_t count() const; //returns the number of nonzeros in the pvector
        void resize(size_t length); //resizes the pvector and overwrites with all zeros
        void from_sparse(const std::vector <entry> &nonzeros, size_t length); //nonzeros[i] is the ith entry
        std::vector <entry> to_sparse(); //returnval[i] is the ith nonzero entry
        void zeros(size_t length); //make the all-zeros vector of specified length, resizing if necessary
        void random(std::mt19937_64 &eng); //fill in the existing vector with random entries
        void random(size_t length, std::mt19937_64 &eng); //make a random vector of given length, resizing if necessary
        std::string to_str() const; //output the pvector in the form of a string (little endian)
        bool from_str(std::string s); //load the pvector from a string of 1s and 0s (little endian)
        bool load_dense(std::string filename); //load pvector from ascii file in dense format (little endian)
        void save_dense(std::string filename); //save pvector to ascii file in dense format (little endian)
        bool load_sparse(std::string filename); //load pvector from ascii file in sparse format
        void save_sparse(std::string filename); //save pvector to ascii file in sparse format
        bool from_sparse_string(std::string s);
        std::string to_sparse_string();
        uint16_t get(size_t i) const; //get entry i
        void set(size_t i, uint16_t val); //set entry i to val
        void operator= (const pvector &V); //overloaded assignment operator
        bool operator== (const pvector &V) const; //overloaded equality operator
        bool operator!= (const pvector &V) const; //overloaded inequality operator
        friend pvector operator+ (const pvector &V1, const pvector &V2); //add vectors mod p
        friend pvector operator- (const pvector &V1, const pvector &V2); //subtract vectors mod p
        pvector &operator+= (const pvector &V); //in place addition mod p
        pvector &operator-= (const pvector &V); //in place addition mod p
        void increment(); //increment as if it were a number
        void from_num(int num, size_t length); //make a bitvector from a number using little-endian convention
        int to_num(); //convert bitvector to a number if length is at most 64. If length > 64 generate error message.
        friend std::ostream& operator<<(std::ostream& os, const pvector& V); //stream bitvector.to_str() e.g. to cout
        friend uint16_t Zp_dot(const pvector &M1, const pvector &V2); //dot product mod p
        friend pvector operator* (const pmatrix &M, const pvector &V); //matrix times vector mod p
        friend pvector operator* (const pvector &V, const pmatrix &M); //vector times matrix mod p
        void append(const pvector &V); //append M to the right of the pvector
        void push_back(uint16_t val); //add one more entry with value val to the end of the vector
        friend pvector hstack(const pvector &V1, const pvector &V2); //append V2 to the right of V1
        pvector range(size_t start, size_t end) const; //output the subvector with entries in the range: start <= j < end
        pmatrix to_row_matrix();
        pmatrix to_col_matrix();
        friend class pmatrix;
        void clear();
    private:
        uint16_t modulus;
        std::vector <uint16_t> entries;
        size_t n; //the length of the vector
};

//four pretty good random numbers for the cost of one
class modprand {
    public:
        modprand(uint16_t p);
        ~modprand();
        uint16_t dist(std::mt19937_64 & eng);
    private:
        uint16_t buf[4];
        uint16_t modulus;
        int supply;
        std::uniform_int_distribution<uint64_t> distr;
        uint64_t mask;
};

#endif