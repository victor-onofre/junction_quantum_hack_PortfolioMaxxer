#include <bit>
#include <fstream>
#include <cstdint>
#include <climits>
#include "modpinverse.hpp"
#include "pmatrix.hpp"
#include "utils.hpp"

modprand::modprand(uint16_t p) {
    modulus = p;
    supply = 0;
    mask = (1ULL<<16) - 1;
}

modprand::~modprand() {
    supply = 0;
}

uint16_t modprand::dist(std::mt19937_64 & eng) {
    if(!supply) {
        uint64_t raw = distr(eng);
        for(size_t i = 0; i < 4; i++) {
            buf[i] = (uint16_t)(raw & mask);
            raw >>= 16;
        }
        supply = 4;
    }
    uint16_t returnval = (uint16_t)(buf[4-supply] % modulus);
    supply--;
    return returnval;
}

bool string_to_pscalar(const std::string &s, uint16_t &val, uint16_t p) {
    int v = stoi(s);
    if(v < 0 || v >= p) {
        notify("Error: bad input in string_to_pscalar.");
        return false;
    }
    val = (uint16_t)v;
    return true;
}

pmatrix::pmatrix() {
    n_cols = 0;
    n_rows = 0;
    modulus = 2;
}

pmatrix::pmatrix(uint16_t p) {
    n_cols = 0;
    n_rows = 0;
    set_modulus(p);
}

pmatrix::pmatrix(size_t rows, size_t cols, uint16_t p) {
    set_modulus(p);
    n_cols = 0;
    n_rows = 0;
    zeros(rows,cols);
}

pmatrix::~pmatrix() {
    n_cols = 0;
    n_rows = 0;
}

void pmatrix::set_modulus(uint16_t p) {
    modulus = p;
    if(p >= PMAX) {
        notify("Error: pmatrix only accommodates modulus below " + std::to_string(PMAX) + ".");
        p = 2; //arbitrary
    }
    if(!p_prime[p]) notify("Error: " + std::to_string(p) + " is not prime.");
}

//returns the modulus
uint16_t pmatrix::p() const {
    return modulus;
}

size_t pmatrix::num_rows() const {
    return n_rows;
}

size_t pmatrix::num_cols() const {
    return n_cols;
}

size_t pmatrix::nonzeros() const {
    size_t count = 0;
    for(const auto & e : entries) if(e) count++;
    return count;
}

void pmatrix::resize(size_t rows, size_t cols) {
    entries.resize(rows*cols);
    fill(entries.begin(), entries.end(), 0);
    n_rows = rows;
    n_cols = cols;
}

void pmatrix::zeros(size_t rows, size_t cols) {
    if(rows != n_rows || cols != n_cols) resize(rows, cols);
    else fill(entries.begin(), entries.end(), 0);
}

void pmatrix::identity(size_t n) {
    zeros(n,n);
    for(size_t row = 0; row < n; row++) set(row,row,1);
}

//add m times row i to row j
void pmatrix::row_add(size_t i, size_t j, uint16_t m) {
    if(i >= n_rows || j >= n_rows) {
        notify("Error: index out of bounds in pmatrix::row_add");
        return;
    }
    uint16_t summand;
    size_t istart = n_cols*i;
    size_t jstart = n_cols*j;
    size_t index_i, index_j;
    for(size_t column_index = 0; column_index < n_cols; column_index++) {
        index_i = istart + column_index;
        index_j = jstart + column_index;
        summand = (m * entries[index_i]) % modulus;
        entries[index_j] = (entries[index_j] + summand) % modulus;
    }
}

//swap row i and row j
void pmatrix::row_swap(size_t i, size_t j) {
    if(i >= n_rows || j >= n_rows) {
        notify("Error: index out of bounds in pmatrix::row_swap");
        return;
    }
    if(i == j) return;
    uint16_t buffer;
    size_t istart = n_cols*i;
    size_t jstart = n_cols*j;
    size_t index_i, index_j;
    for(size_t column_index = 0; column_index < n_cols; column_index++) {
        index_i = istart + column_index;
        index_j = jstart + column_index;
        buffer = entries[index_j];
        entries[index_j] = entries[index_i];
        entries[index_i] = buffer;
    }
}

//add m times column i to column j
void pmatrix::col_add(size_t i, size_t j, uint16_t m) {
    if(i >= n_cols || j >= n_cols) {
        notify("Error: index out of bounds in pmatrix::col_add");
        return;
    }
    uint16_t summand;
    size_t index_i, index_j, start;
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        start = n_cols*row_index;
        index_i = start+i;
        index_j = start+j;
        summand = (m * entries[index_i]) % modulus;
        entries[index_j] = (entries[index_j] + summand) % modulus;
    }
}

//swap column i and column j
void pmatrix::col_swap(size_t i, size_t j) {
    if(i >= n_cols || j >= n_cols) {
        notify("Error: index out of bounds in pmatrix::col_swap");
        return;
    }
    if(i == j) return;
    uint16_t buffer;
    size_t index_i, index_j, start;
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        start = n_cols*row_index;
        index_i = start + i;
        index_j = start + j;
        buffer = entries[index_i];
        entries[index_i] = entries[index_j];
        entries[index_j] = buffer;
    }
}

//elementary row operation: multiply row i by m
void pmatrix::row_mult(size_t i, uint16_t m) {
    if(i >= n_rows) {
        notify("Error: index out of bounds in pmatrix::row_mult");
        return;
    }
    size_t start = i*n_cols;
    size_t end = start + n_cols;
    for(size_t index = start; index < end; index++) entries[index] = (m * entries[index]) % modulus;
}

//elementary col operation: mutiply col i by m
void pmatrix::col_mult(size_t i, uint16_t m) {
    if(i >= n_cols) {
        notify("Error: index out of bounds in pmatrix::col_mult");
        return;
    }
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        size_t index = n_cols*row_index + i;
        entries[index] = (m * entries[index]) % modulus;
    }
}

uint16_t pmatrix::get(size_t row, size_t col) const {
    if(row >= n_rows || col >= n_cols) {
        notify("Error: index out of bounds in pmatrix::get");
        return false;
    }
    return entries[n_cols*row + col];
}

void pmatrix::set(size_t row, size_t col, uint16_t val) {
    if(row >= n_rows || col >= n_cols || val >= modulus) {
        notify("Error: index or value out of bounds in pmatrix::set");
        return;
    }
    entries[n_cols*row + col] = val;
}

//this is not very heavily optimized for performance
std::string pmatrix::to_str() const {
    std::string returnval;
    for(size_t row = 0; row < n_rows; row++) {
        for(size_t col = 0; col < n_cols; col++) {    
            returnval += std::to_string(get(row, col));
            if(col != n_cols - 1) returnval += ",";
        }
        if(row != n_rows - 1) returnval += "\n";
    }
    return returnval;
}

bool pmatrix::from_str(std::string s, char separator) {
    uint16_t val;
    std::vector <std::string> row_strings = tokenize(s, '\n');
    size_t str_rows = row_strings.size();
    if(str_rows == 0) {
        notify("Error: nothing contained in input to pmatrix::from_str");
        return false;
    }
    std::vector<std::vector<std::string> > token_array;
    for(const auto &s : row_strings) {
        token_array.push_back(tokenize(s,separator));
    }
    size_t str_cols = token_array[0].size();
    for(size_t row_index = 1; row_index < token_array.size(); row_index++) {
        if(token_array[row_index].size() != str_cols) {
            notify("Error: uneven row lengths given to pmatrix::from_str");
            return false;
        }
    }
    for(size_t row_index = 0; row_index < str_rows; row_index++) {
        for(size_t col_index = 0; col_index < str_cols; col_index++) {
            if(!string_to_pscalar(token_array[row_index][col_index], val, modulus)) return false;
        }
    }
    resize(str_rows, str_cols);
    for(size_t row_index = 0; row_index < str_rows; row_index++) {
        for(size_t col_index = 0; col_index < str_cols; col_index++) {
            string_to_pscalar(token_array[row_index][col_index], val, modulus);
            set(row_index,col_index,val);
        }
    }
    return true;
}

bool pmatrix::load_dense(std::string filename) {
    std::ifstream infile(filename);
    std::string line;
    std::string noncomments;
    std::string firstline;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    bool first = true;
    while(getline(infile,line)) if(line[0] != '#') {
        if(first) {
            firstline = line;
            first = false;
        }
        else noncomments += line + "\n";
    }
    infile.close();
    int ip, inrows, incols;
    std::vector<std::string> tokens = tokenize(firstline,',');
    if(tokens.size() != 3) {
        notify("Error: misformatted first noncomment line in " + filename);
        return false;
    }
    ip = stoi(tokens[0]);
    inrows = stoi(tokens[1]);
    incols = stoi(tokens[2]);
    if(ip <= 1 || ip >= PMAX) {
        notify("Error: invalid modulus in " + filename);
        return false;
    }
    if(!p_prime[ip]) {
        notify("Error: invalid modulus in " + filename);
        return false;
    }
    set_modulus((uint16_t)ip);
    bool success = from_str(noncomments, ',');
    if(inrows != (int)n_rows) {
        notify("Error: rows claimed does not match rows read in " + filename);
        return false;
    }
    if(incols != (int)n_cols) {
        notify("Error: columns claimed does not match columns read in " + filename);
        return false;
    }
    return success;
}

void pmatrix::save_dense(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << modulus << "," << n_rows << "," << n_cols << std::endl;
    outfile << to_str();
    outfile.close();
}

bool pmatrix::load_sparse(std::string filename) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> firstline_tokens;
    std::vector <std::string> line_tokens;
    std::vector <std::string> pair;
    std::string line;
    int num_rows, num_cols, pmod;
    std::ifstream infile(filename);
    uint16_t val;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            line_tokens = tokenize(line, ',');
            token_array.push_back(line_tokens);
        }
    }
    infile.close();
    if(token_array[0].size() != 3) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    pmod = stoi(token_array[0][0]);
    num_rows = stoi(token_array[0][1]);
    num_cols = stoi(token_array[0][2]);
    if(pmod < 0 || pmod >= PMAX) {
        notify("modulus (" + std::to_string(pmod) +")  is out of range in " + filename);
        return false;
    }
    if(!p_prime[pmod]) {
        notify("modulus (" + std::to_string(pmod) +")  is not prime in " + filename);
        return false;
    }
    modulus = (uint16_t)pmod;
    if(num_rows < 0 || num_rows > 1E6) {
        notify("num_rows(" + std::to_string(num_rows) + ") is out of range in " + filename);
        return false;
    }
    if(num_cols < 0 || num_cols > 1E6) {
        notify("num_cols(" + std::to_string(num_cols) + ")  is out of range in " + filename);
        return 0;
    }
    token_array.erase(token_array.begin()); //pop off the first line
    if(token_array.size() != (size_t)num_rows) {
        notify("Error: number of rows read (" + std::to_string(token_array.size()) + ") doesn't match claim (" + std::to_string(num_rows) +  ").");
        return false;
    }
    zeros(num_rows, num_cols);
    for(size_t row_index = 0; row_index < (size_t)num_rows; row_index++) {
        for(size_t nonzero_index = 0; nonzero_index < token_array[row_index].size(); nonzero_index++) {
            pair = tokenize(token_array[row_index][nonzero_index], ':');
            if(pair.size() != 2) {
                notify("Error: misformatted entry in " + filename);
                return false;
            }
            size_t col_index = std::stoi(pair[0]);
            if(!string_to_pscalar(pair[1], val, modulus)) return false;
            set(row_index, col_index, val);
        }
    }
    infile.close();
    return true;
}

void pmatrix::save_sparse(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << modulus << "," << n_rows << "," << n_cols << std::endl;
    std::vector <std::vector <entry> > nonzeros = to_sparse_rows();
    for(size_t row_index = 0; row_index < nonzeros.size(); row_index++) {
        bool first = true;
        for(size_t one_index = 0; one_index < nonzeros[row_index].size(); one_index++) {
            if(!first) outfile << ",";
            first = false;
            outfile << nonzeros[row_index][one_index].loc << ":" << nonzeros[row_index][one_index].val;
        }
        outfile << std::endl;
    }
    outfile.close();
}

void pmatrix::from_sparse_rows(const std::vector <std::vector <entry> > &nonzeros, size_t num_rows, size_t num_cols) {
    if(nonzeros.size() != num_rows) {
        notify("Error: ones list size not equal to num_rows in pmatrix::from_sparse_rows");
        return;
    }
    zeros(num_rows, num_cols);
    for(size_t row_index = 0; row_index < nonzeros.size(); row_index++) {
        for(const auto &e : nonzeros[row_index]) set(row_index, e.loc, e.val);
    }
}

void pmatrix::from_sparse_cols(const std::vector <std::vector <entry> > &nonzeros, size_t num_rows, size_t num_cols) {
    if(num_cols != nonzeros.size()) {
        notify("Error: ones list size not equal to num_cols in pmatrix::from_sparse_cols");
        return;
    }
    zeros(num_rows, num_cols);
    for(size_t col_index = 0; col_index < nonzeros.size(); col_index++) {
        for(const auto & ent : nonzeros[col_index]) {
            set(ent.loc, col_index, ent.val);
        }
    }
}

void pmatrix::operator= (const pmatrix &M) {
    modulus = M.modulus;
    entries = M.entries;
    n_cols = M.n_cols;
    n_rows = M.n_rows;
}

bool pmatrix::operator== (const pmatrix &M) const {
    if(modulus != M.modulus) return false;
    if(n_cols != M.n_cols) return false;
    if(n_rows != M.n_rows) return false;
    if(entries != M.entries) return false;
    return true;
}

bool pmatrix::operator!= (const pmatrix &M) const {
    return !(*this == M);
}

pmatrix pmatrix::transpose() const {
    pmatrix returnval(modulus);
    returnval.resize(n_cols, n_rows);
    for(size_t row = 0; row < n_rows; row++) {
        for(size_t col = 0; col < n_cols; col++) {
            returnval.set(col, row, get(row,col));
        }
    }
    return returnval;
}

pmatrix operator+ (const pmatrix &M1, const pmatrix &M2) {
    if(M1.n_cols != M2.n_cols || M1.n_rows != M2.n_rows || M1.modulus != M2.modulus) {
        notify("Error: dimension mismatch in pmatrix::+");
        return M1;
    }
    pmatrix returnval(M1.n_rows, M1.n_cols, M1.modulus);
    for(size_t index = 0; index < M1.n_rows*M1.n_cols; index++) {
        returnval.entries[index] = (M1.entries[index] + M2.entries[index]) % M1.modulus;
    }
    return returnval;
}

pmatrix operator- (const pmatrix &M1, const pmatrix &M2) {
    if(M1.n_cols != M2.n_cols || M1.n_rows != M2.n_rows || M1.modulus != M2.modulus) {
        notify("Error: dimension mismatch in pmatrix::-");
        return M1;
    }
    pmatrix returnval(M1.n_rows, M1.n_cols, M1.modulus);
    uint16_t summand;
    for(size_t index = 0; index < M1.n_rows*M1.n_cols; index++) {
        summand = M2.modulus - M2.entries[index]; //won't underflow
        returnval.entries[index] = (M1.entries[index] + summand) % M1.modulus;
    }
    return returnval;
}

pmatrix & pmatrix::operator+= (const pmatrix &M) {
    if(M.n_cols != n_cols || M.n_rows != n_rows || M.modulus != modulus) {
        notify("Error: dimension mismatch in pmatrix::+=");
        return *this;
    }
    for(size_t index = 0; index < n_rows*n_cols; index++) {
        entries[index] = (entries[index] + M.entries[index]) % modulus;
    }
    return *this;
}

pmatrix & pmatrix::operator-= (const pmatrix &M) {
    if(M.n_cols != n_cols || M.n_rows != n_rows || M.modulus != modulus) {
        notify("Error: dimension mismatch in pmatrix::-=");
        return *this;
    }
    uint16_t summand;
    for(size_t index = 0; index < n_rows*n_cols; index++) {
        summand = M.modulus - M.entries[index];
        entries[index] = (entries[index] + summand) % modulus;
    }
    return *this;
}

std::ostream& operator<<(std::ostream &os, const pmatrix &M) {
    os << M.to_str();
    return os;
}

//if we want to make this run faster on sparse matrices, we could
//think of it as a linear combination of rows, and skip the zeros,
//much as we do with the pvector times pmatrix routine 
pmatrix operator* (const pmatrix &M1, const pmatrix &M2) {
    if(M1.n_cols != M2.n_rows || M1.modulus != M2.modulus) {
        notify("Error: dimension mismatch in pmatrix::*");
        return M1;
    }
    size_t i,j,k;
    std::vector<int> IM1(M1.n_rows*M1.n_cols, 0);
    std::vector<int> IM2(M2.n_rows*M2.n_cols, 0);
    std::vector<int> IM3(M1.n_rows*M2.n_cols, 0);
    for(i = 0; i < M1.n_rows*M1.n_cols; i++) IM1[i] = (int)M1.entries[i];
    for(i = 0; i < M2.n_rows*M2.n_cols; i++) IM2[i] = (int)M2.entries[i];
    for(i = 0; i < M1.n_rows; i++) {
        for(j = 0; j < M1.n_cols; j++) {
            for(k = 0; k < M2.n_cols; k++) {
                IM3[i*M2.n_cols + k] += M1.entries[i*M1.n_cols + j] * M2.entries[j*M2.n_cols + k];
            }
        }
    }
    pmatrix returnval(M1.n_rows, M2.n_cols, M1.modulus);
    for(i = 0; i < M1.n_rows*M2.n_cols; i++) returnval.entries[i] = (uint16_t)(IM3[i] % M1.modulus);
    return returnval;
}

pvector operator* (const pmatrix &M, const pvector &V) {
    if(M.num_cols() != V.size() || M.modulus != V.modulus) {
        notify("Error: dimension mismatch in matrix times vector");
        return V;
    }
    std::vector<int> isum(M.num_rows(), 0);
    pvector returnval(M.num_rows(), M.p());
    for(size_t j = 0; j < V.size(); j++) {
        if(V.get(j)) {
            for(size_t i = 0; i < M.num_rows(); i++) {
                isum[i] += V.get(j) * M.get(i,j);
            }
        }
    }
    for(size_t i = 0; i < M.num_rows(); i++) returnval.set(i, isum[i] % returnval.p());
    return returnval;
}

pvector operator* (const pvector &V, const pmatrix &M) {
    if(M.num_rows() != V.size() || M.modulus != V.modulus) {
        notify("Error: dimension mismatch in vector times matrix");
        return V;
    }
    std::vector<int> isum(M.num_cols(), 0);
    pvector returnval(M.num_cols(), M.p());
    for(size_t i = 0; i < M.num_rows(); i++) {
        if(V.get(i) != 0) {
            for(size_t j = 0; j < M.num_cols(); j++) {
                isum[j] += V.get(i)*M.get(i,j);
            }
        }
    }
    for(size_t j = 0; j < M.num_cols(); j++) returnval.set(j, isum[j] % returnval.p());
    return returnval; 
}

void pmatrix::random(std::mt19937_64 &eng) {
    modprand gen(modulus);
    for(size_t index = 0; index < n_rows*n_cols; index++) {
        entries[index] = gen.dist(eng);
    }
}

void pmatrix::random(size_t rows, size_t cols, std::mt19937_64 &eng) {
    resize(rows, cols);
    random(eng);
}

std::vector <std::vector <entry> > pmatrix::to_sparse_rows() {
    std::vector <std::vector <entry> > nonzeros;
    entry e;
    uint16_t val;
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        std::vector <entry> rowvec;
        for(size_t col_index = 0; col_index < n_cols; col_index++) {
            val = get(row_index,col_index);
            if(val != 0) {
                e.loc = col_index;
                e.val = val;
                rowvec.push_back(e);
            }
        }
        nonzeros.push_back(rowvec);
    }
    return nonzeros;
}

std::vector <std::vector <entry> > pmatrix::to_sparse_cols() {
    std::vector <std::vector <entry> > nonzeros;
    entry e;
    uint16_t val;
    for(size_t col_index = 0; col_index < n_cols; col_index++) {
        std::vector <entry> colvec;
        for(size_t row_index = 0; row_index < n_rows; row_index++) {
            val = get(row_index, col_index);
            if(val != 0) {
                e.loc = row_index;
                e.val = val;
                colvec.push_back(e);
            }
        }
        nonzeros.push_back(colvec);
    }
    return nonzeros;
}

uint16_t Zp_dot(const pmatrix &M1, const pmatrix &M2) {
    int count = 0;
    if(M1.modulus != M2.modulus) {
        notify("Error: modulus mismatch in pmatrix dot product");
        return 0;
    }
    if(M1.n_rows == 1 && M2.n_rows == 1) { //row vector dot row vector
        if(M1.n_cols != M2.n_cols) {
            notify("Error: dimension mismatch in pmatrix dot product");
            return 0;
        }
        for(size_t index = 0; index < M1.n_cols; index++) {
            count += (M1.entries[index] * (int) M2.entries[index]) % M2.modulus;
        }
        return (uint16_t)(count % M2.modulus);
    }
    if(M1.n_rows == 1 && M2.n_cols == 1) { //row vector dot column vector
        if(M1.n_cols != M2.n_rows) {
            notify("Error: dimension mismatch in pmatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t index = 0; index < M1.n_cols; index++) {
            count += (M1.get(0,index) * M2.get(index,0)) % M2.modulus;
        }
        return (uint16_t)(count % M2.modulus);
    }
    if(M1.n_cols == 1 && M2.n_rows == 1) { //column vector dot row vector
        if(M1.n_rows != M2.n_cols) {
            notify("Error: dimension mismatch in pmatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t index = 0; index < M1.n_rows; index++) {
            count += (M1.get(index,0) * M2.get(0, index)) % M2.modulus;
        }
        return (uint16_t)(count % M2.modulus);
    }
    if(M1.n_cols == 1 && M2.n_cols == 1) { //column vector dot column vector
        if(M1.n_rows != M2.n_rows) {
            notify("Error: dimension mismatch in pmatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t index = 0; index < M1.n_rows; index++) {
            count += (M1.get(index,0) * M2.get(index,0)) % M2.modulus;
        }
        return (uint16_t)(count % M2.modulus);
    }
    notify("Error: cannot take dot product of non-vector pmatrices");
    return 0;
}

//the columns from start to end-1
pmatrix pmatrix::col_range(size_t start, size_t end) const {
    if(start >= end || start < 0 || end > n_cols) {
        notify("Invalid range in pmatrix::col_range");
        return *this;
    }
    pmatrix returnval(n_rows, end-start, modulus);
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        for(size_t col_index = 0; col_index < end-start; col_index++) {
            returnval.set(row_index, col_index, get(row_index,start+col_index));
        }
    }
    return returnval;
}

//the rows from start to end-1
pmatrix pmatrix::row_range(size_t start, size_t end) const {
    if(start >= end || start < 0 || end > n_rows) {
        notify("Invalid range in pmatrix::row_range");
        return *this;
    }
    pmatrix returnval(end-start, n_cols, modulus);
    for(size_t row_index = 0; row_index < end-start; row_index++) {
        for(size_t col_index = 0; col_index < n_cols; col_index++) {
            returnval.set(row_index, col_index, get(start+row_index, col_index));
        }
    }
    return returnval;
}

pmatrix pmatrix::row(size_t i) const {
    pmatrix returnval(1, n_cols, modulus);
    if(i >= n_rows) {
        notify("Error: index out of range in pmatrix::row");
        return returnval;
    }
    for(size_t index = 0; index < n_cols; index++) returnval.set(0,index,get(i,index));
    return returnval;
}

pmatrix pmatrix::col(size_t j) const {
    pmatrix returnval(n_rows, 1, modulus);
    if(j >= n_cols) {
        notify("Error: index out of range in pmatrix::col");
        return returnval;
    }
    for(size_t index = 0; index < n_rows; index++) returnval.set(index,0,get(index,j));
    return returnval;
}

pmatrix pmatrix::col_transpose(size_t j) const {
    pmatrix returnval(1, n_rows, modulus);
    if(j >= n_cols) {
        notify("Error: index out of range in pmatrix::col_transpose");
        return returnval;
    }
    for(size_t index = 0; index < n_rows; index++) returnval.set(0,index,get(index,j));
    return returnval;
}

pvector pmatrix::row_vec(size_t i) const {
    pvector returnval(n_cols, modulus);
    if(i >= n_rows) {
        notify("Error: index out of range in pmatrix::row_vec");
        return returnval;
    }
    for(size_t j = 0; j < n_cols; j++) returnval.set(j,get(i,j));
    return returnval;
}

pvector pmatrix::col_vec(size_t j) const {
    pvector returnval(n_rows, modulus);
    if(j >= n_cols) {
        notify("Error: index out of range in pmatrix::col_vec");
        return returnval;
    }
    for(size_t i = 0; i < n_rows; i++) returnval.set(i,get(i,j));
    return returnval;
}

void pmatrix::append_right(const pmatrix &M) {
    if(n_cols == 0) { //empty pmatrix
        *this = M;
        return;
    }
    if(M.n_rows != n_rows || M.modulus != modulus) {
        notify("Error: dimension mismatch in pmatrix::append_right");
        return;
    }
    pmatrix tmp = *this;
    resize(tmp.num_rows(), tmp.num_cols() + M.num_cols());
    for(size_t i = 0; i < n_rows; i++) {
        for(size_t j = 0; j < tmp.num_cols(); j++) set(i,j,tmp.get(i,j));
        for(size_t j = 0; j < M.num_cols(); j++) set(i,tmp.num_cols()+j,M.get(i,j));
    }
}

void pmatrix::append_right(const pvector &V) {
    if(n_cols == 0) { //empty pmatrix
        modulus = V.modulus;
        n_rows = V.size();
        n_cols = 1;
        entries = V.entries;
        return;
    }
    if(V.size() != n_rows) {
        notify("Error: dimension mismatch in pmatrix::append_right");
        return;
    }
    pmatrix tmp = *this;
    resize(tmp.num_rows(), tmp.num_cols() + 1);
    for(size_t i = 0; i < n_rows; i++) {
        for(size_t j = 0; j < tmp.num_cols(); j++) set(i,j,tmp.get(i,j));
        set(i,tmp.num_cols(),V.get(i));
    }
}

void pmatrix::append_below(const pmatrix &M) {
    if(n_rows == 0) { //empty pmatrix
        *this = M;
        return;
    }
    if(M.n_cols != n_cols) {
        notify("Error: dimension mismatch in pmatrix::append_below");
        return;
    }
    entries.insert( entries.end(), M.entries.begin(), M.entries.end() );
    n_rows += M.n_rows;
}

void pmatrix::append_below(const pvector &V) {
    if(n_cols == 0) { //empty pmatrix
        modulus = V.modulus;
        n_rows = 1;
        n_cols = V.size();
        entries = V.entries;
        return;
    }
    if(V.size() != n_cols) {
        notify("Error: dimension mismatch in pmatrix::append_below");
        return;
    }
    entries.insert( entries.end(), V.entries.begin(), V.entries.end() );
    n_rows++;
}

pmatrix vstack(const pmatrix &M1, const pmatrix &M2) {
    pmatrix returnval = M1;
    returnval.append_below(M2);
    return returnval;
}

pmatrix hstack(const pmatrix &M1, const pmatrix &M2) {
    pmatrix returnval = M1;
    returnval.append_right(M2);
    return returnval;
}

uint16_t pmatrix::determinant() const {
    if(n_rows != n_cols) {
        notify("Error: non-square matrix does not have determinant");
        return 0;
    }
    pmatrix M = *this;
    int sign = 1;
    for(size_t col_index = 0; col_index < n_cols; col_index++) {
        size_t pivot = col_index;
        while(!(M.get(pivot,col_index))) {
            pivot++;
            if(pivot >= n_rows) return 0; //the column zeros all the way to the bottom
        }
        if(pivot != col_index) {
            M.row_swap(pivot,col_index);
            sign *= -1;
        }
        for(size_t row_index = 0; row_index < n_rows; row_index++) {
            if(row_index != col_index && M.get(row_index, col_index)) {
                uint16_t minus_target = modulus - M.get(row_index,col_index);
                uint16_t inverse_pivot = p_inverse[modulus][M.get(col_index,col_index)];
                uint16_t coefficient = (minus_target * inverse_pivot) % modulus;
                M.row_add(col_index, row_index, coefficient);
            }
        }
    }
    uint16_t det = 1;
    for(size_t index = 0; index < n_cols; index++) det = (det * M.get(index,index)) % modulus;
    if(sign == -1) det = modulus - det;
    return det;
}

//reduces M in-place to reduced row echelon form, using only the columns 0..cmax-1
//returns rank
size_t rref(pmatrix &M, size_t cmax, uint16_t modulus) {
    if(cmax > M.num_cols()) {
        notify("Error: cmax > num_cols in pmatrix rref");
        return 0;
    }
    bool skip;
    size_t row_index = 0;
    size_t col_index = 0;
    size_t rank = 0;
    while(row_index < M.num_rows() && col_index < cmax) {
        size_t pivot = row_index;
        skip = false;
        while(!(M.get(pivot,col_index))) {
            pivot++;
            if(pivot >= M.num_rows()) {
                skip = true; //the column has zeros all the way to the bottom
                break;
            }
        }
        if(!skip) {
            rank++;
            if(pivot != row_index) {
                M.row_swap(pivot,row_index);
            }
            uint16_t pivot_val = M.get(row_index, col_index);
            if(pivot_val != 1) M.row_mult(row_index, p_inverse[modulus][pivot_val]);
            for(size_t r = 0; r < M.num_rows(); r++) {
                if(r != row_index && M.get(r, col_index)) {
                    uint16_t minus_target = modulus - M.get(r,col_index);
                    M.row_add(row_index, r, minus_target);
                }
            }
            row_index++;
        }
        col_index++;
    }
    return rank;
}

//reduces M in-place to row echelon form (not reduced), using only columns 0..cmax-1
//returns rank
size_t ref(pmatrix &M, size_t cmax, uint16_t modulus) {
    if(cmax > M.num_cols()) {
        notify("Error: cmax > num_cols in pmatrix ref");
        return 0;
    }
    bool skip;
    size_t row_index = 0;
    size_t col_index = 0;
    size_t rank = 0;
    while(row_index < M.num_rows() && col_index < cmax) {
        size_t pivot = row_index;
        skip = false;
        while(!(M.get(pivot,col_index))) {
            pivot++;
            if(pivot >= M.num_rows()) {
                skip = true; //the column has zeros all the way to the bottom
                break;
            }
        }
        if(!skip) {
            rank++;
            if(pivot != row_index) {
                M.row_swap(pivot,row_index);
            }
            uint16_t pivot_val = M.get(row_index, col_index);
            if(pivot_val != 1) M.row_mult(row_index, p_inverse[modulus][pivot_val]);
            for(size_t r = row_index+1; r < M.num_rows(); r++) {
                uint16_t minus_target = modulus - M.get(r,col_index);
                M.row_add(row_index, r, minus_target);
            }
            row_index++;
        }
        col_index++;
    }
    return rank;
}

pmatrix pmatrix::row_echelon_form() const {
    pmatrix returnval = *this;
    ref(returnval, returnval.num_cols(), modulus);
    return returnval;
}

pmatrix pmatrix::reduced_row_echelon_form() const {
    pmatrix returnval = *this;
    rref(returnval, returnval.num_cols(), modulus);
    return returnval;
}

//GM = R. R is reduced row echelon form. G is invertible.
void pmatrix::reduced_row_echelon_decomp(pmatrix &G, pmatrix &R) const {
    R = *this;
    pmatrix I(modulus);
    I.identity(n_rows);
    R.append_right(I);
    rref(R, n_cols, modulus);
    G = R.col_range(n_cols, R.num_cols());
    R = R.col_range(0,n_cols);
}

//GM = R. R is row echelon form. G is invertible.
void pmatrix::row_echelon_decomp(pmatrix &G, pmatrix &R) const {
    R = *this;
    pmatrix I(modulus);
    I.identity(n_rows);
    R.append_right(I);
    ref(R, n_cols, modulus);
    G = R.col_range(n_cols, R.num_cols());
    R = R.col_range(0,n_cols);
}

int pmatrix::rank() const {
    pmatrix tmp = *this;
    return ref(tmp, n_cols, modulus);
}

pmatrix pmatrix::inverse(bool &invertible) const {
    if(n_rows != n_cols) {
        notify("Error: non-square matrix cannot be inverted");
        invertible = false;
        return *this;
    }
    pmatrix id(modulus);
    pmatrix augmented(modulus);
    id.identity(n_rows);
    augmented = hstack(*this, id);
    size_t rank = rref(augmented, n_rows, modulus);
    if(rank == n_rows) invertible = true;
    else invertible = false;
    return augmented.col_range(n_cols,2*n_cols);
}

pmatrix operator* (const pmatrix &M, const uint16_t &s) {
    pmatrix returnval(M.num_rows(), M.num_cols(), M.p());
    for(size_t i = 0; i < M.num_rows(); i++) {
        for(size_t j = 0; j < M.num_cols(); j++) {
            uint16_t val = (M.get(i,j) * s) % M.p();
            returnval.set(i,j,val);
        }
    }
    return returnval;
}

pmatrix operator* (const uint16_t &s, const pmatrix &M) {
    return M * s;
}

void pmatrix::clear() {
    n_cols = 0;
    n_rows = 0;
    entries.clear();
}

//------------------------------------------------------------------------------------------------------

pvector::pvector() {
    modulus = 0;
    n = 0;
}

void pvector::set_modulus(uint16_t p) {
    modulus = p;
    if(p >= PMAX) {
        notify("Error: pvector only accommodates modulus below " + std::to_string(PMAX) + ".");
        p = 2; //arbitrary
    }
    if(!p_prime[p]) notify("Error: " + std::to_string(p) + " is not prime.");
}

pvector::pvector(uint16_t p) {
    set_modulus(p);
    n = 0;
}

pvector::pvector(size_t length, uint16_t p) {
    set_modulus(p);
    n = 0;
    zeros(length);
}

pvector::~pvector() {
    n = 0;
}

size_t pvector::size() const {
    return n;
}

size_t pvector::nonzeros() const {
    size_t count = 0;
    for(const auto & e : entries) if(e) count++;
    return count;
}

uint16_t pvector::p() const {
    return modulus;
}

size_t pvector::count() const {
    size_t count = 0;
    for(size_t index = 0; index < n; index++) if(entries[index]) count++;
    return count;
}

void pvector::resize(size_t length) {
    n = length;
    entries.resize(length);
    fill(entries.begin(), entries.end(), 0);
}

void pvector::zeros(size_t length) {
    if(length != n) resize(length);
    else fill(entries.begin(), entries.end(), 0);
}

uint16_t pvector::get(size_t i) const {
    if(i >= n) {
        notify("Error: index out of bounds in pvector::get");
        return false;
    }
    return entries[i];
}

void pvector::set(size_t i, uint16_t val) {
    if(i >= n || val >= modulus) {
        notify("Error: index or value out of bounds in pvector::set");
        return;
    }
    entries[i] = val;
}

void pvector::increment() {
    size_t index = 0;
    while(index < n) {
        if(entries[index] < modulus-1) {
            (entries[index])++;
            break;
        }
        else {
            entries[index] = 0;
            index++;
        }
    }
}

void pvector::from_num(int num, size_t length) {
    zeros(length);
    if(num < 0) {
        notify("Error: negative input to from_num");
        return;
    }
    size_t index = 0;
    uint16_t val;
    while(num > 0 && index < length) {
        val = num % modulus;
        entries[index++] = val;
        num /= modulus;
    }
}

int pvector::to_num() {
    if(pow(modulus, n) > (double)INT_MAX) {
        notify("Error: pvector of length " + std::to_string(n) + " cannot be converted to int");
        return 0;
    }
    int place_value = 1;
    int num = 0;
    for(size_t index = 0; index < n; index++) {
        num += place_value * entries[index];
        place_value *= modulus;
    }
    return num;
}

//this is not very heavily optimized for performance
std::string pvector::to_str() const {
    std::string returnval;
    for(size_t i = 0; i < n; i++) {
        returnval += std::to_string(get(i));
        if(i < n-1) returnval += ",";
    }
    return returnval;
}

bool pvector::from_str(std::string s) {
    if(s.length() == 0) {
        notify("Error: empty string in pvector::from_str");
        return false;
    }
    std::vector <std::string> tokens = tokenize(s, ',');
    resize(tokens.size());
    for(size_t i = 0; i < tokens.size(); i++) {
        int val = stoi(tokens[i]);
        if(val < 0 || val > modulus) {
            notify("Error: invalid input to pvector::from_str");
            return false;
        }
        entries[i] = (uint16_t)val;
    }
    return true;
}

bool pvector::load_dense(std::string filename) {
    std::ifstream infile(filename);
    std::string line;
    std::vector <std::string> noncomments;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            noncomments.push_back(line);
        }
    }
    if(noncomments.size() > 2) {
        notify("Error: more than two noncomment lines in " + filename);
        return false;
    }
    infile.close();
    int ip, in;
    std::vector<std::string> tokens = tokenize(noncomments[0],',');
    if(tokens.size() != 2) {
        notify("Error: misformatted first noncomment line in " + filename);
        return false;
    }
    ip = stoi(tokens[0]);
    in = stoi(tokens[1]);
    if(ip <= 1 || ip >= PMAX) {
        notify("Error: invalid modulus in " + filename);
        return false;
    }
    if(!p_prime[ip]) {
        notify("Error: invalid modulus in " + filename);
        return false;
    }
    set_modulus((uint16_t)ip);
    bool success = from_str(noncomments[1]);
    if((int)n != in) {
        notify("Error: number of symbols read does not match claim in " + filename);
        return false;
    }
    return success;
}

void pvector::save_dense(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << modulus << "," << n << std::endl;
    outfile << to_str();
    outfile.close();
}

bool pvector::from_sparse_string(std::string s) {
    std::vector<std::string> token_array = tokenize(s,',');
    std::vector <std::string> pair;
    zeros(n);
    for(size_t nonzero_index = 0; nonzero_index < token_array.size(); nonzero_index++) {
        pair = tokenize(token_array[nonzero_index], ':');
        if(pair.size() != 2) {
            notify("Error: misformatted entry");
            return false;
        }
        size_t index = std::stoi(pair[0]);
        uint16_t val;
        if(!string_to_pscalar(pair[1], val, modulus)) return false;
        set(index, val);
    }
    return true;
}

std::string pvector::to_sparse_string() {
    std::string str;
    std::vector <entry> sp = to_sparse();
    bool first = true;
    for(const auto &e : sp) {
        if(!first) str += ',';
        str += std::to_string((int)e.loc) + ':' + std::to_string((int)e.val);
        first = false;
    }
    return str;
}

bool pvector::load_sparse(std::string filename) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> line_tokens;
    std::vector <std::string> pair;
    std::string line;
    int ni, pmod;
    std::ifstream infile(filename);
    uint16_t val;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            line_tokens = tokenize(line, ',');
            token_array.push_back(line_tokens);
        }
    }
    infile.close();
    if(token_array[0].size() != 2) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    pmod = stoi(token_array[0][0]);
    ni = stoi(token_array[0][1]);
    if(pmod < 0 || pmod >= PMAX) {
        notify("modulus (" + std::to_string(pmod) +")  is out of range in " + filename);
        return false;
    }
    if(!p_prime[pmod]) {
        notify("modulus (" + std::to_string(pmod) +")  is not prime in " + filename);
        return false;
    }
    modulus = (uint16_t)pmod;
    if(ni < 0 || ni > 1E6) {
        notify("num_cols(" + std::to_string(ni) + ")  is out of range in " + filename);
        return 0;
    }
    token_array.erase(token_array.begin()); //pop off the first line
    zeros((size_t)ni);
    for(size_t nonzero_index = 0; nonzero_index < token_array[0].size(); nonzero_index++) {
        pair = tokenize(token_array[0][nonzero_index], ':');
        if(pair.size() != 2) {
            notify("Error: misformatted entry in " + filename);
            return false;
        }
        size_t index = std::stoi(pair[0]);
        if(!string_to_pscalar(pair[1], val, modulus)) return false;
        set(index, val);
    }
    infile.close();
    return true;
}

std::vector <entry> pvector::to_sparse() {
    std::vector <std::vector <entry> > nonzeros;
    entry e;
    uint16_t val;
    std::vector <entry> sparsevec;
    for(size_t index = 0; index < n; index++) {
        val = entries[index];
        if(val != 0) {
            e.loc = index;
            e.val = val;
            sparsevec.push_back(e);
        }
    }
    return sparsevec;
}

void pvector::save_sparse(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << modulus << "," << n << std::endl;
    std::vector <entry> nonzeros = to_sparse();
    bool first = true;
    for(size_t one_index = 0; one_index < nonzeros.size(); one_index++) {
        if(!first) outfile << ",";
        first = false;
        outfile << nonzeros[one_index].loc << ":" << nonzeros[one_index].val;
    }
    outfile << std::endl;
    outfile.close();
}

void pvector::from_sparse(const std::vector <entry> &nonzeros, size_t length) {
    zeros(length);
    for(const auto &e : nonzeros) set(e.loc, e.val);
}

void pvector::operator= (const pvector &V) { 
    modulus = V.modulus;
    n = V.n;
    entries = V.entries;
}

bool pvector::operator== (const pvector &V) const {
    if(n != V.n) return false;
    if(modulus != V.modulus) return false;
    for(size_t index = 0; index < n; index++) {
        if(entries[index] != V.entries[index]) return false;
    }
    return true;
}

bool pvector::operator!= (const pvector &V) const {
    return !(*this == V);
}

pvector operator+ (const pvector &V1, const pvector &V2) {
    if(V1.n != V2.n) {
        notify("Error: dimension mismatch in pvector::+");
        return V1;
    }
    if(V1.modulus != V2.modulus) {
        notify("Error: modulus mismatch in pvector::+");
        return V1;
    }
    pvector returnval = V1;
    for(size_t index = 0; index < V1.n; index++) {
        returnval.entries[index] = (returnval.entries[index] + V2.entries[index])% returnval.modulus;
    }
    return returnval;
}

pvector operator- (const pvector &V1, const pvector &V2) {
    if(V1.n != V2.n) {
        notify("Error: dimension mismatch in pvector::-");
        return V1;
    }
    if(V1.modulus != V2.modulus) {
        notify("Error: modulus mismatch in pvector::-");
        return V1;
    }
    pvector returnval(V1.n, V1.modulus);
    uint16_t summand;
    for(size_t index = 0; index < V1.n; index++) {
        summand = V2.modulus - V2.entries[index]; //won't underflow
        returnval.entries[index] = (V1.entries[index] + summand) % V1.modulus;
    }
    return returnval;
}

pvector & pvector::operator+= (const pvector &V) {
    if(V.n != n) {
        notify("Error: dimension mismatch in pvector::+=");
        return *this;
    }
    if(V.modulus != modulus) {
        notify("Error: modulus mismatch in pvector::+=");
        return *this;
    }
    for(size_t index = 0; index < n; index++) {
        entries[index] = (entries[index] + V.entries[index]) % modulus;
    }
    return *this;
}

pvector & pvector::operator-= (const pvector &V) {
    if(V.n != n) {
        notify("Error: dimension mismatch in pvector::-=");
        return *this;
    }
    if(V.modulus != modulus) {
        notify("Error: modulus mismatch in pvector::-=");
        return *this;
    }
    uint16_t summand;
    for(size_t index = 0; index < n; index++) {
        summand = V.modulus - V.entries[index];
        entries[index] = (entries[index] + summand) % modulus;
    }
    return *this;
}

std::ostream& operator<<(std::ostream &os, const pvector &V) {
    os << V.to_str();
    return os;
}

void pvector::random(std::mt19937_64 &eng) {
    modprand gen(modulus);
    for(size_t index = 0; index < n; index++) {
        entries[index] = gen.dist(eng);
    }
}

void pvector::random(size_t length, std::mt19937_64 &eng) {
    resize(length);
    random(eng);
}

uint16_t Zp_dot(const pvector &V1, const pvector &V2) {
    if(V1.n != V2.n || V1.modulus != V2.modulus) {
        notify("Error: dimension mismatch in pvector dot product");
        return 0;
    }
    int count = 0;
    for(size_t index = 0; index < V1.n; index++) {
        count += (V1.entries[index] * V2.entries[index]) % V1.modulus;
    }
    return (uint16_t)(count % V1.modulus);
}

//the columns from start to end-1
pvector pvector::range(size_t start, size_t end) const {
    if(start >= end || start < 0 || end > n) {
        notify("Invalid range in pvector::col_range");
        return *this;
    }
    pvector returnval(end-start, modulus);
    for(size_t i = 0; i < end-start; i++) {
        returnval.set(i, get(start+i));
    }
    return returnval;
}

void pvector::append(const pvector &V) {
    pvector tmp = *this;
    resize(tmp.size() + V.size());
    for(size_t j = 0; j < tmp.size(); j++) set(j,tmp.get(j));
    for(size_t j = 0; j < V.size(); j++) set(tmp.size()+j,V.get(j));
}

void pvector::push_back(uint16_t val) {
    n++;
    entries.push_back(val);
}

pmatrix pvector::to_row_matrix() {
    pmatrix returnval(1,n,modulus);
    returnval.entries = entries;
    return returnval;
}

pmatrix pvector::to_col_matrix() {
    pmatrix returnval(n,1,modulus);
    returnval.entries = entries;
    return returnval;
}

pvector hstack(const pvector &V1, const pvector &V2) {
    pvector returnval = V1;
    returnval.append(V2);
    return returnval;
}

pvector operator* (const pvector &V, const uint16_t &s) {
    pvector returnval(V.size(), V.p());
    for(size_t index = 0; index < V.size(); index++) {
        uint16_t val = ( V.get(index) * s ) % V.p();
        returnval.set(index, val);
    }
    return returnval;
}

pvector operator* (const uint16_t &s, const pvector &V) {
    return V * s;
}

void pvector::clear() {
    n = 0;
    entries.clear();
}
