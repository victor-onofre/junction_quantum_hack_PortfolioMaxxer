#include <bit>
#include <fstream>
#include <cstdint>
#include "bitmatrix.hpp"
#include "utils.hpp"

bitmatrix::bitmatrix() {
    n_cols = 0;
    n_rows = 0;
    n_blocks = 0;
    blocks_per_row = 0;
}

bitmatrix::bitmatrix(size_t rows, size_t cols) {
    n_cols = 0;
    n_rows = 0;
    n_blocks = 0;
    blocks_per_row = 0;
    zeros(rows,cols);
}

bitmatrix::~bitmatrix() {
    n_cols = 0;
    n_rows = 0;
    n_blocks = 0;
    blocks_per_row = 0;
}

size_t bitmatrix::num_rows() const {
    return n_rows;
}

size_t bitmatrix::num_cols() const {
    return n_cols;
}

size_t bitmatrix::count_ones() const {
    size_t count = 0;
    for(size_t index = 0; index < n_blocks; index++) {
        count += std::popcount(blocks[index]);
    }
    return count;
}

void bitmatrix::resize(size_t rows, size_t cols) {
    blocks_per_row = cols/64;
    if(cols%64) blocks_per_row++;
    n_blocks = blocks_per_row*rows;
    blocks.resize(n_blocks);
    fill(blocks.begin(), blocks.end(), 0);
    n_rows = rows;
    n_cols = cols;
}

void bitmatrix::zeros(size_t rows, size_t cols) {
    if(rows != n_rows || cols != n_cols) resize(rows, cols);
    else fill(blocks.begin(), blocks.end(), 0);
}

void bitmatrix::identity(size_t n) {
    zeros(n,n);
    for(size_t row = 0; row < n; row++) set(row,row,1);
}

//add row i to row j
void bitmatrix::row_add(size_t i, size_t j) {
    if(i >= n_rows || j >= n_rows) {
        notify("Error: index out of bounds in bitmatrix::row_add");
        return;
    }
    size_t ri = blocks_per_row*i;
    size_t rj = blocks_per_row*j;
    for(size_t b = 0; b < blocks_per_row; b++) {
        blocks[rj+b] ^= blocks[ri+b];
    }
}

//swap row i and row j
void bitmatrix::row_swap(size_t i, size_t j) {
    if(i >= n_rows || j >= n_rows) {
        notify("Error: index out of bounds in bitmatrix::row_swap");
        return;
    }
    if(i == j) return;
    size_t ri = blocks_per_row*i;
    size_t rj = blocks_per_row*j;
    uint64_t buffer;
    for(size_t b = 0; b < blocks_per_row; b++) {
        buffer = blocks[rj+b];
        blocks[rj+b] = blocks[ri+b];
        blocks[ri+b] = buffer;
    }
}

//add column i to column j
void bitmatrix::col_add(size_t i, size_t j) {
    if(i >= n_cols || j >= n_cols) {
        notify("Error: index out of bounds in bitmatrix::col_add");
        return;
    }
    size_t imask_index = i%64;
    size_t iblock_index = i/64;
    size_t jmask_index = j%64;
    size_t jblock_index = j/64;
    uint64_t imask = 1ULL<<imask_index;
    size_t shiftleft = 0;
    size_t shiftright = 0;
    if(jmask_index > imask_index) shiftleft = jmask_index - imask_index;
    if(jmask_index < imask_index) shiftright = imask_index - jmask_index;
    uint64_t val;
    size_t r;
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        r = blocks_per_row*row_index;
        val = imask & blocks[r+iblock_index];
        //It's probably quicker for the chip to do this, with one (and sometimes both)
        //of the values zero than to use an if statement.
        val <<= shiftleft;
        val >>= shiftright;
        blocks[r+jblock_index] ^= val;
    }
}

//swap column i and column j
void bitmatrix::col_swap(size_t i, size_t j) {
    if(i >= n_cols || j >= n_cols) {
        notify("Error: index out of bounds in bitmatrix::col_swap");
        return;
    }
    if(i == j) return;
    //this is less transparent than a naive approach with get and set
    //hopefully it pays off in performance. should test.
    size_t imask_index = i%64;
    size_t iblock_index = i/64;
    size_t jmask_index = j%64;
    size_t jblock_index = j/64;
    uint64_t imask = 1ULL<<imask_index;
    uint64_t jmask = 1ULL<<jmask_index;
    size_t shiftleft = 0;
    size_t shiftright = 0;
    if(jmask_index > imask_index) shiftleft = jmask_index - imask_index;
    if(jmask_index < imask_index) shiftright = imask_index - jmask_index;
    uint64_t val;
    size_t r;
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        r = blocks_per_row*row_index;
        //three CNOTs makes a SWAP
        val = imask & blocks[r+iblock_index];
        val <<= shiftleft;
        val >>= shiftright;
        blocks[r+jblock_index] ^= val;

        val = jmask & blocks[r+jblock_index];
        val >>= shiftleft;  //need to go backwards
        val <<= shiftright; //need to go backwards
        blocks[r+iblock_index] ^= val;

        val = imask & blocks[r+iblock_index];
        val <<= shiftleft;
        val >>= shiftright;
        blocks[r+jblock_index] ^= val;
    }
}

void bitmatrix::row_zero(size_t i) {
    if(i >= n_rows) {
        notify("Error: index out of bounds in bitmatrix::row_zero");
        return;
    }
    size_t start = i*blocks_per_row;
    size_t stop = start+blocks_per_row;
    for(size_t j = start; j < stop; j++) blocks[j] = 0;
}

void bitmatrix::col_zero(size_t i) {
    if(i >= n_cols) {
        notify("Error: index out of bounds in bitmatrix::col_zero");
        return;
    }
    size_t block_offset = i/64;
    size_t bit_offset = i%64;
    uint64_t mask = ~(1ULL<<bit_offset);
    for(size_t j = 0; j < n_rows; j++) {
        blocks[block_offset+j*blocks_per_row] &= mask;
    }
}

bool bitmatrix::get(size_t row, size_t col) const {
    if(row >= n_rows || col >= n_cols) {
        notify("Error: index out of bounds in bitmatrix::get");
        return false;
    }
    size_t block_index = row*blocks_per_row + col/64;
    size_t bit_index = col%64;
    uint64_t bm = 1ULL<<bit_index;
    return (bool)(bm & blocks[block_index]);
}

void bitmatrix::set(size_t row, size_t col, bool val) {
    if(row >= n_rows || col >= n_cols) {
        notify("Error: index out of bounds in bitmatrix::set");
        return;
    }
    size_t block_index = row*blocks_per_row + col/64;
    size_t bit_index = col%64;
    uint64_t bm = 1ULL<<bit_index;
    if(val) blocks[block_index] |= bm;
    else blocks[block_index] &= ~bm;
}

void bitmatrix::flip(size_t row, size_t col) {
    if(row >= n_rows || col >= n_cols) {
        notify("Error: index out of bounds in bitmatrix::flip");
        return;
    }
    size_t block_index = row*blocks_per_row + col/64;
    size_t bit_index = col%64;
    uint64_t bm = 1ULL<<bit_index;
    blocks[block_index] ^= bm;
}

//this is not very heavily optimized for performance
std::string bitmatrix::to_str() const {
    std::string returnval;
    for(size_t row = 0; row < n_rows; row++) {
        for(size_t col = 0; col < n_cols; col++) {    
            if(get(row, col)) returnval.push_back('1');
            else returnval.push_back('0');
        }
        if(row != n_rows - 1) returnval.push_back('\n');
    }
    return returnval;
}

bool bitmatrix::from_str(std::string s, char separator) {
    std::vector <std::string> row_strings = tokenize(s, separator);
    size_t str_rows = row_strings.size();
    if(str_rows == 0) {
        notify("Error: nothing contained in input to bitmatrix::from_str");
        return false;
    }
    size_t str_cols = row_strings[0].size();
    for(size_t row_index = 0; row_index < str_rows; row_index++) {
        if(row_strings[row_index].size() != str_cols) {
            notify("Error: uneven row lengths given to bitmatrix::from_str");
            return false;
        }
    }
    for(size_t row_index = 0; row_index < str_rows; row_index++) {
        for(size_t col_index = 0; col_index < str_cols; col_index++) {
            if(row_strings[row_index][col_index] != '0' && row_strings[row_index][col_index] != '1') {
                notify("Error: non-bit character given to bitmatrix::from_str");
                return false;
            }
        }
    }
    resize(str_rows, str_cols);
    for(size_t row_index = 0; row_index < str_rows; row_index++) {
        for(size_t col_index = 0; col_index < str_cols; col_index++) {
            if(row_strings[row_index][col_index] == '0') set(row_index,col_index,0);
            if(row_strings[row_index][col_index] == '1') set(row_index,col_index,1);
        }
    }
    return true;
}

bool bitmatrix::load_dense(std::string filename) {
    std::ifstream infile(filename);
    std::string line;
    std::vector <std::string> noncomments;
    size_t columns, line_index, char_index;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            noncomments.push_back(line);
        }
    }
    columns = noncomments[0].length();
    for(line_index = 1; line_index < noncomments.size(); line_index++) {
        if(noncomments[line_index].length() != columns) {
            notify("Error: mismatched line lengths in " + filename);
            return false;
        }
    }
    for(line_index = 0; line_index < noncomments.size(); line_index++) {
        for(char_index = 0; char_index < noncomments[line_index].length(); char_index++) {
            if(noncomments[line_index][char_index] != '0' && noncomments[line_index][char_index] != '1') {
                notify("Error: invalid (non-bit) character in " + filename);
                return false;
            }
        }
    }
    zeros(noncomments.size(), columns);
    for(line_index = 0; line_index < noncomments.size(); line_index++) {
        for(char_index = 0; char_index < noncomments[line_index].length(); char_index++) {
            if(noncomments[line_index][char_index] == '1') set(line_index, char_index, 1);
        }
    }
    infile.close();
    return true;
}

void bitmatrix::save_dense(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << to_str();
    outfile.close();
}

bool bitmatrix::load_sparse(std::string filename) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> firstline_tokens;
    std::vector <std::string> line_tokens;
    std::string line;
    size_t num_rows, num_cols;
    std::ifstream infile(filename);
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            line_tokens = tokenize(line, '\t');
            token_array.push_back(line_tokens);
        }
    }
    infile.close();
    if(token_array[0].size() != 2) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    num_rows = stoi(token_array[0][0]);
    num_cols = stoi(token_array[0][1]);
    if(num_rows < 0 || num_rows > 1E6) {
        notify("num_rows(" + std::to_string(num_rows) + ") is out of range in " + filename);
        return false;
    }
    if(num_cols < 0 || num_cols > 1E6) {
        notify("num_cols(" + std::to_string(num_cols) + ")  is out of range in " + filename);
        return 0;
    }
    token_array.erase(token_array.begin()); //pop off the first line
    if(token_array.size() != num_rows) {
        notify("Error: number of rows read (" + std::to_string(token_array.size()) + ") doesn't match claim (" + std::to_string(num_rows) +  ").");
        return false;
    }
    zeros(num_rows, num_cols);
    for(size_t row_index = 0; row_index < num_rows; row_index++) {
        for(size_t one_index = 0; one_index < token_array[row_index].size(); one_index++) {
            size_t col_index = stoi(token_array[row_index][one_index]);
            if(col_index >= num_cols) {
                notify("Error: column index " + std::to_string(col_index) + " out of range in " + filename);
                return false;
            }
            set(row_index, col_index, 1);
        }
    }
    infile.close();
    return true;
}

void bitmatrix::save_sparse(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << n_rows << "\t" << n_cols << std::endl;
    std::vector <std::vector <size_t> > nonzeros = to_sparse_rows();
    for(size_t row_index = 0; row_index < nonzeros.size(); row_index++) {
        bool first = true;
        for(size_t one_index = 0; one_index < nonzeros[row_index].size(); one_index++) {
            if(!first) outfile << "\t";
            first = false;
            outfile << nonzeros[row_index][one_index];
        }
        outfile << std::endl;
    }
    outfile.close();
}

void bitmatrix::from_sparse_rows(const std::vector <std::vector <size_t> > &ones, size_t num_rows, size_t num_cols) {
    if(ones.size() != num_rows) {
        notify("Error: ones list size not equal to num_rows in bitmatrix::from_sparse_rows");
        return;
    }
    zeros(num_rows, num_cols);
    for(size_t row_index = 0; row_index < ones.size(); row_index++) {
        for(size_t col_index : ones[row_index]) {
            set(row_index, col_index, true);
        }
    }
}

void bitmatrix::from_sparse_cols(const std::vector <std::vector <size_t> > &ones, size_t num_rows, size_t num_cols) {
    if(num_cols != ones.size()) {
        notify("Error: ones list size not equal to num_cols in bitmatrix::from_sparse_cols");
        return;
    }
    zeros(num_rows, num_cols);
    for(size_t col_index = 0; col_index < ones.size(); col_index++) {
        for(size_t row_index : ones[col_index]) {
            set(row_index, col_index, true);
        }
    }
}

void bitmatrix::operator= (const bitmatrix &M) { 
    blocks = M.blocks;
    n_blocks = M.n_blocks;
    n_rows = M.n_rows;
    n_cols = M.n_cols;
    blocks_per_row = M.blocks_per_row;
}

bool bitmatrix::operator== (const bitmatrix &M) const {
    if(n_cols != M.n_cols) return false;
    if(n_rows != M.n_rows) return false;
    for(size_t block_index = 0; block_index < blocks.size(); block_index++) {
        if(blocks[block_index] != M.blocks[block_index]) return false;
    }
    return true;
}

bool bitmatrix::operator!= (const bitmatrix &M) const {
    return !(*this == M);
}

bitmatrix bitmatrix::transpose() const {
    bitmatrix returnval;
    returnval.resize(n_cols, n_rows);
    for(size_t row = 0; row < n_rows; row++) {
        for(size_t col = 0; col < n_cols; col++) {
            returnval.set(col, row, get(row,col));
        }
    }
    return returnval;
}

bitmatrix operator+ (const bitmatrix &M1, const bitmatrix &M2) {
    if(M1.n_cols != M2.n_cols || M1.n_rows != M2.n_rows) {
        notify("Error: dimension mismatch in bitmatrix::+");
        return M1;
    }
    bitmatrix returnval = M1;
    for(size_t block_index = 0; block_index < M1.n_blocks; block_index++) {
        returnval.blocks[block_index] ^= M2.blocks[block_index];
    }
    return returnval;
}

bitmatrix & bitmatrix::operator+= (const bitmatrix &M) {
    if(M.n_cols != n_cols || M.n_rows != n_rows) {
        notify("Error: dimension mismatch in bitmatrix::+=");
        return *this;
    }
    for(size_t block_index = 0; block_index < n_blocks; block_index++) {
        blocks[block_index] ^= M.blocks[block_index];
    }
    return *this;
}

std::ostream& operator<<(std::ostream &os, const bitmatrix &M) {
    os << M.to_str();
    return os;
}

bitmatrix operator* (const bitmatrix &M1, const bitmatrix &M2) {
    if(M1.n_cols != M2.n_rows) {
        notify("Error: dimension mismatch in bitmatrix::*");
        return M1;
    }
    bitmatrix returnval(M1.n_rows, M2.n_cols);
    for(size_t i = 0; i < M1.n_rows; i++) {
        for(size_t j = 0; j < M1.n_cols; j++) {
            if(M1.get(i,j)) {
                for(size_t k = 0; k < returnval.blocks_per_row; k++) {
                    returnval.blocks[returnval.blocks_per_row*i+k] ^= M2.blocks[M2.blocks_per_row*j+k];
                }
            }
        }
    }
    return returnval;
}

bitvector operator* (const bitmatrix &M, const bitvector &V) {
    if(M.num_cols() != V.size()) {
        notify("Error: dimension mismatch in matrix times vector");
        return V;
    }
    bitvector returnval(M.num_rows());
    for(size_t i = 0; i < M.num_rows(); i++) {
        returnval.set(i, Z2_dot(V,M.row_vec(i)));
    }
    return returnval;
}

bitvector operator* (const bitvector &V, const bitmatrix &M) {
    if(M.num_rows() != V.size()) {
        notify("Error: dimension mismatch in vector times matrix");
        return V;
    }
    bitvector returnval(M.num_cols());
    for(size_t i = 0; i < M.num_rows(); i++) {
        if(V.get(i)) returnval += M.row_vec(i);
    }
    return returnval; 
}

bitmatrix operator& (const bitmatrix &M1, const bitmatrix &M2) {
    if(M1.n_cols != M2.n_cols || M1.n_rows != M2.n_rows) {
        notify("Error: dimension mismatch in bitmatrix::&");
        return M1;
    }
    bitmatrix returnval = M1;
    for(size_t block_index = 0; block_index < M1.n_blocks; block_index++) {
        returnval.blocks[block_index] &= M2.blocks[block_index];
    }
    return returnval;
}

bitmatrix operator| (const bitmatrix &M1, const bitmatrix &M2) {
    if(M1.n_cols != M2.n_cols || M1.n_rows != M2.n_rows) {
        notify("Error: dimension mismatch in bitmatrix::|");
        return M1;
    }
    bitmatrix returnval = M1;
    for(size_t block_index = 0; block_index < M1.n_blocks; block_index++) {
        returnval.blocks[block_index] |= M2.blocks[block_index];
    }
    return returnval;
}

bitmatrix bitmatrix::operator~ () const {
    bitmatrix returnval(n_rows, n_cols);
    for(size_t block_index = 0; block_index < n_blocks; block_index++) {
        returnval.blocks[block_index] = ~blocks[block_index];
    }
    if(n_cols%64) {
        uint64_t mask = (1ULL<<(n_cols%64))-1;
        for(size_t row_index = 0; row_index < n_rows; row_index++) {
            returnval.blocks[(row_index+1)*blocks_per_row-1] &= mask;
        }
    }
    return returnval;
}

void bitmatrix::random(std::mt19937_64 &eng) {
    std::uniform_int_distribution<uint64_t> distr;
    for(size_t block_index = 0; block_index < n_blocks; block_index++) {
        blocks[block_index] = distr(eng);
    }
    if(n_cols%64) {
        uint64_t mask = (1ULL<<(n_cols%64))-1;
        for(size_t row_index = 0; row_index < n_rows; row_index++) {
            blocks[(row_index+1)*blocks_per_row-1] &= mask;
        }
    }
}

void bitmatrix::random(size_t rows, size_t cols, std::mt19937_64 &eng) {
    resize(rows, cols);
    random(eng);
}

std::vector <std::vector <size_t> > bitmatrix::to_sparse_rows() {
    std::vector <std::vector <size_t> > ones;
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        std::vector <size_t> rowvec;
        for(size_t col_index = 0; col_index < n_cols; col_index++) {
            if(get(row_index,col_index)) rowvec.push_back(col_index);
        }
        ones.push_back(rowvec);
    }
    return ones;
}

std::vector <std::vector <size_t> > bitmatrix::to_sparse_cols() {
    std::vector <std::vector <size_t> > ones;
    for(size_t col_index = 0; col_index < n_cols; col_index++) {
        std::vector <size_t> colvec;
        for(size_t row_index = 0; row_index < n_rows; row_index++) {
            if(get(row_index,col_index)) colvec.push_back(row_index);
        }
        ones.push_back(colvec);
    }
    return ones;
}

bool Z2_dot(const bitmatrix &M1, const bitmatrix &M2) {
    return (bool)(Z_dot(M1,M2)%2);
}

size_t Z_dot(const bitmatrix &M1, const bitmatrix &M2) {
    if(M1.n_rows == 1 && M2.n_rows == 1) { //row vector dot row vector
        if(M1.n_cols != M2.n_cols) {
            notify("Error: dimension mismatch in bitmatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t block_index = 0; block_index < M1.n_blocks; block_index++) {
            count += std::popcount(M1.blocks[block_index] & M2.blocks[block_index]);
        }
        return count;
    }
    if(M1.n_rows == 1 && M2.n_cols == 1) { //row vector dot column vector
        if(M1.n_cols != M2.n_rows) {
            notify("Error: dimension mismatch in bitmatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t index = 0; index < M1.n_cols; index++) {
            count += (int)(M1.get(0,index) && M2.get(index,0));
        }
        return count;
    }
    if(M1.n_cols == 1 && M2.n_rows == 1) { //column vector dot row vector
        if(M1.n_rows != M2.n_cols) {
            notify("Error: dimension mismatch in bimatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t index = 0; index < M1.n_rows; index++) {
            count += (int)(M1.get(index,0) && M2.get(0, index));
        }
        return count;
    }
    if(M1.n_cols == 1 && M2.n_cols == 1) { //column vector dot column vector
        if(M1.n_rows != M2.n_rows) {
            notify("Error: dimension mismatch in bitmatrix dot product");
            return 0;
        }
        size_t count = 0;
        for(size_t index = 0; index < M1.n_rows; index++) {
            count += (int)(M1.get(index,0) && M2.get(index,0));
        }
        return count;
    }
    notify("Error: cannot take dot product of non-vector bitmatrices");
    return 0;
}

//the columns from start to end-1
bitmatrix bitmatrix::col_range(size_t start, size_t end) const {
    if(start >= end || start < 0 || end > n_cols) {
        notify("Invalid range in bitmatrix::col_range");
        return *this;
    }
    bitmatrix returnval(n_rows, end-start);
    for(size_t row_index = 0; row_index < n_rows; row_index++) {
        for(size_t col_index = 0; col_index < end-start; col_index++) {
            returnval.set(row_index, col_index, get(row_index,start+col_index));
        }
    }
    return returnval;
}

//the rows from start to end-1
bitmatrix bitmatrix::row_range(size_t start, size_t end) const {
    if(start >= end || start < 0 || end > n_rows) {
        notify("Invalid range in bitmatrix::row_range");
        return *this;
    }
    bitmatrix returnval(end-start, n_cols);
    for(size_t row_index = 0; row_index < end-start; row_index++) {
        for(size_t block_index = 0; block_index < blocks_per_row; block_index++) {
            returnval.blocks[blocks_per_row*row_index + block_index] = blocks[blocks_per_row*(row_index+start)+block_index];
        }
    }
    return returnval;
}

bitmatrix bitmatrix::row(size_t i) const {
    bitmatrix returnval(1, n_cols);
    if(i >= n_rows) {
        notify("Error: index out of range in bitmatrix::row");
        return returnval;
    }
    size_t start_block = blocks_per_row*i;
    for(size_t block_index = 0; block_index < blocks_per_row; block_index++) {
        returnval.blocks[block_index] = blocks[start_block+block_index];
    }
    return returnval;
}

bitmatrix bitmatrix::col(size_t j) const {
    bitmatrix returnval(n_rows, 1);
    if(j >= n_cols) {
        notify("Error: index out of range in bitmatrix::col");
        return returnval;
    }
    size_t shift = j%64;
    uint64_t mask = 1ULL<<shift;
    size_t block_offset = j/64;
    uint64_t val;
    for(size_t i = 0; i < n_rows; i++) {
        val = (blocks[blocks_per_row*i+block_offset] & mask)>>shift;
        returnval.blocks[i] = val;
    }
    return returnval;
}

bitmatrix bitmatrix::col_transpose(size_t j) const {
    bitmatrix returnval(1, n_rows);
    if(j >= n_cols) {
        notify("Error: index out of range in bitmatrix::col_transpose");
        return returnval;
    }
    size_t shift = j%64;
    uint64_t mask = 1ULL<<shift;
    size_t block_offset = j/64;
    uint64_t val;
    for(size_t i = 0; i < n_rows; i++) {
        val = (blocks[blocks_per_row*i+block_offset] & mask)>>shift;
        val <<= (i%64);
        returnval.blocks[i/64] ^= val;
    }
    return returnval;
}

bitvector bitmatrix::row_vec(size_t i) const {
    bitvector returnval(n_cols);
    if(i >= n_rows) {
        notify("Error: index out of range in bitmatrix::row_vec");
        return returnval;
    }
    size_t offset = blocks_per_row*i;
    for(size_t i = 0; i < blocks_per_row; i++) returnval.blocks[i] = blocks[i+offset];
    return returnval;
}

bitvector bitmatrix::col_vec(size_t i) const {
    bitvector returnval(n_rows);
    if(i >= n_cols) {
        notify("Error: index out of range in bitmatrix::col_vec");
        return returnval;
    }
    bitmatrix tmp = col_transpose(i);
    returnval.blocks = tmp.blocks;
    return returnval;
}

void bitmatrix::append_right(const bitmatrix &M) {
    if(n_cols == 0) {
        *this = M;
        return;
    }
    if(M.n_rows != n_rows) {
        notify("Error: dimension mismatch in bitmatrix::append_right");
        return;
    }
    bitmatrix tmp = *this;
    resize(tmp.num_rows(), tmp.num_cols() + M.num_cols());
    for(size_t i = 0; i < n_rows; i++) {
        for(size_t j = 0; j < tmp.num_cols(); j++) set(i,j,tmp.get(i,j));
        for(size_t j = 0; j < M.num_cols(); j++) set(i,tmp.num_cols()+j,M.get(i,j));
    }
}

void bitmatrix::append_below(const bitmatrix &M) {
    if(n_rows == 0) {
        *this = M;
        return;
    }
    if(M.n_cols != n_cols) {
        notify("Error: dimension mismatch in bitmatrix::append_below");
        return;
    }
    blocks.insert( blocks.end(), M.blocks.begin(), M.blocks.end() );
    n_rows += M.n_rows;
    n_blocks += M.n_blocks;
}

void bitmatrix::append_right(const bitvector &V) {
    if(n_cols == 0) {
        resize(V.n_bits,1);
        for(size_t i = 0; i < V.size(); i++) set(i,0,V.get(i));
        return;
    }
    if(V.n_bits != n_rows) {
        notify("Error: dimension mismatch in bitmatrix::append_right");
        return;
    }
    bitmatrix tmp = *this;
    resize(tmp.num_rows(), tmp.num_cols() + 1);
    for(size_t i = 0; i < n_rows; i++) {
        for(size_t j = 0; j < tmp.num_cols(); j++) set(i,j,tmp.get(i,j));
        set(i,tmp.num_cols(),V.get(i));
    }
}

void bitmatrix::append_below(const bitvector &V) {
    if(n_rows == 0) {
        resize(1,V.n_bits);
        blocks = V.blocks;
        return;
    }
    if(V.n_bits != n_cols) {
        notify("Error: dimension mismatch in bitmatrix::append_below");
        return;
    }
    blocks.insert( blocks.end(), V.blocks.begin(), V.blocks.end() );
    n_rows += 1;
    n_blocks += V.n_blocks;
}

bitmatrix vstack(const bitmatrix &M1, const bitmatrix &M2) {
    bitmatrix returnval = M1;
    returnval.append_below(M2);
    return returnval;
}

bitmatrix hstack(const bitmatrix &M1, const bitmatrix &M2) {
    bitmatrix returnval = M1;
    returnval.append_right(M2);
    return returnval;
}

bool bitmatrix::determinant() const {
    if(n_rows != n_cols) {
        notify("Error: non-square matrix does not have determinant");
        return false;
    }
    bitmatrix M = *this;
    for(size_t col_index = 0; col_index < n_cols; col_index++) {
        size_t pivot = col_index;
        while(!(M.get(pivot,col_index))) {
            pivot++;
            if(pivot >= n_rows) return false; //the column zeros all the way to the bottom
        }
        if(pivot != col_index) {
            M.row_swap(pivot,col_index);
        }
        for(size_t row_index = 0; row_index < n_rows; row_index++) {
            if(row_index != col_index && M.get(row_index, col_index)) {
                M.row_add(col_index, row_index);
            }
        }
    }
    return true;
}

//reduces M in-place to reduced row echelon form, using only the columns 0..cmax-1
//returns rank
size_t rref(bitmatrix &M, size_t cmax) {
    if(cmax > M.num_cols()) {
        notify("Error: cmax > num_cols in rref");
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
            for(size_t r = 0; r < M.num_rows(); r++) {
                if(r != row_index && M.get(r, col_index)) {
                    M.row_add(row_index, r);
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
size_t ref(bitmatrix &M, size_t cmax) {
    if(cmax > M.num_cols()) {
        notify("Error: cmax > num_cols in ref");
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
            for(size_t r = row_index+1; r < M.num_rows(); r++) {
                if(M.get(r, col_index)) {
                    M.row_add(row_index, r);
                }
            }
            row_index++;
        }
        col_index++;
    }
    return rank;
}

bitmatrix bitmatrix::row_echelon_form() const {
    bitmatrix returnval = *this;
    ref(returnval, returnval.num_cols());
    return returnval;
}

bitmatrix bitmatrix::reduced_row_echelon_form() const {
    bitmatrix returnval = *this;
    rref(returnval, returnval.num_cols());
    return returnval;
}

//GM = R. R is reduced row echelon form. G is invertible.
void bitmatrix::reduced_row_echelon_decomp(bitmatrix &G, bitmatrix &R) const {
    R = *this;
    bitmatrix I;
    I.identity(n_rows);
    R.append_right(I);
    rref(R, n_cols);
    G = R.col_range(n_cols, R.num_cols());
    R = R.col_range(0,n_cols);
}

//GM = R. R is row echelon form. G is invertible.
void bitmatrix::row_echelon_decomp(bitmatrix &G, bitmatrix &R) const {
    R = *this;
    bitmatrix I;
    I.identity(n_rows);
    R.append_right(I);
    ref(R, n_cols);
    G = R.col_range(n_cols, R.num_cols());
    R = R.col_range(0,n_cols);
}

int bitmatrix::rank() const {
    bitmatrix tmp = *this;
    return ref(tmp, n_cols);
}

bitmatrix bitmatrix::inverse(bool &invertible) const {
    if(n_rows != n_cols) {
        notify("Error: non-square matrix cannot be inverted");
        invertible = false;
        return *this;
    }
    bitmatrix id;
    bitmatrix augmented;
    id.identity(n_rows);
    augmented = hstack(*this, id);
    size_t rank = rref(augmented, n_rows);
    if(rank == n_rows) invertible = true;
    else invertible = false;
    return augmented.col_range(n_cols,2*n_cols);
}

bitmatrix operator* (const bitmatrix &M, const bool &s) {
    if(s) return M;
    bitmatrix returnval(M.num_rows(), M.num_cols());
    return returnval;
}

bitmatrix operator* (const bool &s, const bitmatrix &M) {
    return M * s;
}

//------------------------------------------------------------------------------------------------------

bitvector::bitvector() {
    n_bits = 0;
    n_blocks = 0;
}

bitvector::bitvector(size_t length) {
    n_bits = 0;
    n_blocks = 0;
    zeros(length);
}

bitvector::~bitvector() {
    n_bits = 0;
    n_blocks = 0;
}

size_t bitvector::size() const {
    return n_bits;
}

size_t bitvector::count() const {
    size_t count = 0;
    for(size_t index = 0; index < n_blocks; index++) {
        count += std::popcount(blocks[index]);
    }
    return count;
}

void bitvector::resize(size_t length) {
    n_blocks = length/64;
    if(length%64) n_blocks++;
    blocks.resize(n_blocks);
    fill(blocks.begin(), blocks.end(), 0);
    n_bits = length;
}

void bitvector::zeros(size_t length) {
    if(length != n_bits) resize(length);
    else fill(blocks.begin(), blocks.end(), 0);
}

bool bitvector::get(size_t i) const {
    if(i >= n_bits) {
        notify("Error: index out of bounds in bitvector::get");
        return false;
    }
    size_t block_index = i/64;
    size_t bit_index = i%64;
    uint64_t bm = 1ULL<<bit_index;
    return (bool)(bm & blocks[block_index]);
}

void bitvector::set(size_t i, bool val) {
    if(i >= n_bits) {
        notify("Error: index out of bounds in bitvector::set");
        return;
    }
    size_t block_index = i/64;
    size_t bit_index = i%64;
    uint64_t bm = 1ULL<<bit_index;
    if(val) blocks[block_index] |= bm;
    else blocks[block_index] &= ~bm;
}

void bitvector::flip(size_t i) {
    if(i >= n_bits) {
        notify("Error: index out of bounds in bitvector::flip");
        return;
    }
    size_t block_index = i/64;
    size_t bit_index = i%64;
    uint64_t bm = 1ULL<<bit_index;
    blocks[block_index] ^= bm;
}

void bitvector::increment() {
    size_t block_index = 0;
    while(block_index < n_blocks) {
        if(blocks[block_index] < UINT64_MAX) {
            (blocks[block_index])++;
            break;
        }
        else {
            blocks[block_index] = 0;
            block_index++;
        }
    }
    if(n_bits%64) {
        uint64_t mask = (1ULL<<(n_bits%64))-1;
        blocks[n_blocks-1] &= mask;
    }
}

void bitvector::from_num(uint64_t num, size_t length) {
    zeros(length);
    uint64_t mask = (1ULL<<length) - 1;
    blocks[0] = num;
    blocks[0] &= mask;
}

uint64_t bitvector::to_num() {
    if(n_bits > 64) notify("Error: bitvector of length " + std::to_string(n_bits) + " cannot be converted to uint64_t");
    return blocks[0];
}

//this is not very heavily optimized for performance
std::string bitvector::to_str() const {
    std::string returnval;
    for(size_t i = 0; i < n_bits; i++) {    
        if(get(i)) returnval.push_back('1');
        else returnval.push_back('0');
    }
    return returnval;
}

bool bitvector::from_str(std::string s) {
    if(s.length() == 0) {
        notify("Error: empty string in bitvector::from_str");
        return false;
    }
    for(size_t i = 0; i < s.length(); i++) {
        if(s[i] != '0' && s[i] != '1') {
            notify("Error: invalid character in bitvector::from_str");
            return false;
        }
    }
    resize(s.length());
    for(size_t i = 0; i < s.length(); i++) {
        if(s[i] == '1') set(i,1);
    }
    return true;
}

bool bitvector::load_dense(std::string filename) {
    std::ifstream infile(filename);
    std::string line;
    std::vector <std::string> noncomments;
    size_t char_index;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            noncomments.push_back(line);
        }
    }
    if(noncomments.size() > 1) {
        notify("Error: more than one noncomment line in " + filename);
        return false;
    }
    for(size_t i = 0; i < noncomments[0].length(); i++) {
        if(noncomments[0][i] != '0' && noncomments[0][i] != '1') {
            notify("Error: invalid character in bitvector::load_dense");
            return false;
        }
    }
    zeros(noncomments[0].size());
    for(char_index = 0; char_index < noncomments[0].length(); char_index++) {
        if(noncomments[0][char_index] == '1') set(char_index, 1);
    }
    infile.close();
    return true;
}

void bitvector::save_dense(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << to_str();
    outfile.close();
}

bool bitvector::load_sparse(std::string filename) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> firstline_tokens;
    std::vector <std::string> line_tokens;
    std::string line;
    size_t num_bits;
    std::ifstream infile(filename);
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            line_tokens = tokenize(line, '\t');
            token_array.push_back(line_tokens);
        }
    }
    infile.close();
    if(token_array[0].size() != 1) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    num_bits = stoi(token_array[0][0]);
    if(num_bits < 0 || num_bits > 1E6) {
        notify("num_rows(" + std::to_string(num_bits) + ") is out of range in " + filename);
        return false;
    }
    token_array.erase(token_array.begin()); //pop off the first line
    if(token_array.size() != 1) {
        notify("Error: more than one noncomment row in sparse vector file " + filename);
        return false;
    }
    zeros(num_bits);
    for(size_t one_index = 0; one_index < token_array[0].size(); one_index++) {
        size_t bit_index = stoi(token_array[0][one_index]);
        if(bit_index >= num_bits) {
                notify("Error: bit index " + std::to_string(bit_index) + " out of range in " + filename);
                return false;
        }
        set(bit_index, 1);
    }
    infile.close();
    return true;
}

std::vector <size_t> bitvector::to_sparse() {
    std::vector <size_t> ones;
    for(size_t i = 0; i < n_bits; i++) {
        if(get(i)) ones.push_back(i);
    }
    return ones;
}

void bitvector::save_sparse(std::string filename) {
    std::ofstream outfile;
    outfile.open(filename);
    outfile << n_bits << std::endl;
    std::vector <size_t> nonzeros = to_sparse();
    bool first = true;
    for(size_t one_index = 0; one_index < nonzeros.size(); one_index++) {
        if(!first) outfile << "\t";
        first = false;
        outfile << nonzeros[one_index];
    }
    outfile << std::endl;
    outfile.close();
}

void bitvector::from_sparse(const std::vector <size_t> &ones, size_t length) {
    zeros(length);
    for(size_t i : ones) set(i, 1);
}

void bitvector::operator= (const bitvector &V) { 
    blocks = V.blocks;
    n_blocks = V.n_blocks;
    n_bits = V.n_bits;
}

bool bitvector::operator== (const bitvector &V) const {
    if(n_bits != V.n_bits) return false;
    for(size_t block_index = 0; block_index < blocks.size(); block_index++) {
        if(blocks[block_index] != V.blocks[block_index]) return false;
    }
    return true;
}

bool bitvector::operator!= (const bitvector &V) const {
    return !(*this == V);
}

bitvector operator+ (const bitvector &V1, const bitvector &V2) {
    if(V1.n_bits != V2.n_bits) {
        notify("Error: dimension mismatch in bitvector::+");
        return V1;
    }
    bitvector returnval = V1;
    for(size_t block_index = 0; block_index < V1.n_blocks; block_index++) {
        returnval.blocks[block_index] ^= V2.blocks[block_index];
    }
    return returnval;
}

bitvector & bitvector::operator+= (const bitvector &V) {
    if(V.n_bits != n_bits) {
        notify("Error: dimension mismatch in bitvector::+=");
        return *this;
    }
    for(size_t block_index = 0; block_index < n_blocks; block_index++) {
        blocks[block_index] ^= V.blocks[block_index];
    }
    return *this;
}

std::ostream& operator<<(std::ostream &os, const bitvector &V) {
    os << V.to_str();
    return os;
}

bitvector operator& (const bitvector &V1, const bitvector &V2) {
    if(V1.n_bits != V2.n_bits) {
        notify("Error: dimension mismatch in bitvector::&");
        return V1;
    }
    bitvector returnval = V1;
    for(size_t block_index = 0; block_index < V1.n_blocks; block_index++) {
        returnval.blocks[block_index] &= V2.blocks[block_index];
    }
    return returnval;
}

bitvector operator| (const bitvector &V1, const bitvector &V2) {
    if(V1.n_bits != V2.n_bits) {
        notify("Error: dimension mismatch in bitvector::|");
        return V1;
    }
    bitvector returnval = V1;
    for(size_t block_index = 0; block_index < V1.n_blocks; block_index++) {
        returnval.blocks[block_index] |= V2.blocks[block_index];
    }
    return returnval;
}

bitvector bitvector::operator~ () const {
    bitvector returnval(n_bits);
    for(size_t block_index = 0; block_index < n_blocks; block_index++) {
        returnval.blocks[block_index] = ~blocks[block_index];
    }
    if(n_bits%64) {
        uint64_t mask = (1ULL<<(n_bits%64))-1;
        returnval.blocks[n_blocks-1] &= mask;
    }
    return returnval;
}

void bitvector::random(std::mt19937_64 &eng) {
    std::uniform_int_distribution<uint64_t> distr;
    for(size_t block_index = 0; block_index < n_blocks; block_index++) {
        blocks[block_index] = distr(eng);
    }
    if(n_bits%64) {
        uint64_t mask = (1ULL<<(n_bits%64))-1;
        blocks[n_blocks-1] &= mask;
    }
}

void bitvector::random(size_t length, std::mt19937_64 &eng) {
    resize(length);
    random(eng);
}

bool Z2_dot(const bitvector &V1, const bitvector &V2) {
    return (bool)(Z_dot(V1,V2)%2);
}

size_t Z_dot(const bitvector &V1, const bitvector &V2) {
    if(V1.n_bits != V2.n_bits) {
        notify("Error: dimension mismatch in bitvector dot product");
        return 0;
    }
    size_t count = 0;
    for(size_t block_index = 0; block_index < V1.n_blocks; block_index++) {
        count += std::popcount(V1.blocks[block_index] & V2.blocks[block_index]);
    }
    return count;
}

//the columns from start to end-1
bitvector bitvector::range(size_t start, size_t end) const {
    if(start >= end || start < 0 || end > n_bits) {
        notify("Invalid range in bitvector::col_range");
        return *this;
    }
    bitvector returnval(end-start);
    for(size_t i = 0; i < end-start; i++) {
        returnval.set(i, get(start+i));
    }
    return returnval;
}

void bitvector::append(const bitvector &V) {
    bitvector tmp = *this;
    resize(tmp.size() + V.size());
    for(size_t j = 0; j < tmp.size(); j++) set(j,tmp.get(j));
    for(size_t j = 0; j < V.size(); j++) set(tmp.size()+j,V.get(j));
}

void bitvector::push_back(bool val) {
    if(n_bits%64 == 0) {
        blocks.push_back((uint64_t)val);
        n_bits++;
        n_blocks++;
    }
    else {
        n_bits++;
        set(n_bits-1, val);
    }
}

bitmatrix bitvector::to_row_matrix() {
    bitmatrix returnval(1,n_bits);
    returnval.blocks = blocks;
    return returnval;
}

bitmatrix bitvector::to_col_matrix() {
    bitmatrix returnval(n_bits,1);
    for(size_t i = 0; i < n_bits; i++) returnval.set(i,0,get(i));
    return returnval;
}

bitvector bitvector::complement() const {
    bitvector returnval(n_bits);
    for(size_t block = 0; block < returnval.blocks.size(); block++) returnval.blocks[block] = ~blocks[block];
    if(n_bits%64) {
        uint64_t mask = (1ULL<<(n_bits%64))-1;
        returnval.blocks[n_blocks-1] &= mask;
    }
    return returnval;
}

bitvector hstack(const bitvector &V1, const bitvector &V2) {
    bitvector returnval = V1;
    returnval.append(V2);
    return returnval;
}

bitvector operator* (const bitvector &V, const bool &s) {
    if(s) return V;
    bitvector returnval(V.size()); //all zeros
    return returnval;
}

bitvector operator* (const bool &s, const bitvector &V) {
    return V * s;
}

bitmatrix permute_rows(const bitmatrix &M, const std::vector<size_t> &permutation) {
    bitmatrix R = M.row(permutation[0]);
    for(size_t i = 1; i < M.num_rows(); i++) R.append_below(M.row(permutation[i]));
    return R;
}
//This is much faster than directly permuting the columns
bitmatrix permute_columns(const bitmatrix &M, const std::vector<size_t> &permutation) {
    bitmatrix MT = M.transpose();
    MT = permute_rows(MT, permutation);
    return MT.transpose();
}

bitvector permute_bits(const bitvector &v, const std::vector<size_t> &permutation) {
    bitvector r(v.size());
    for(size_t i = 0; i < v.size(); i++) r.set(i, v.get(permutation[i]));
    return r;
}


size_t new_weight(const bitmatrix &B, const bitvector &v, size_t i) {
    size_t weight = 0;
    size_t start_block = i*B.blocks_per_row;
    for(size_t index = 0; index < B.blocks_per_row; index++) {
        weight += std::popcount(B.blocks[start_block + index] ^ v.blocks[index]);
    }
    return weight;
}

bitvector rand_weight_k(std::mt19937_64 &eng, size_t blocksize, size_t error_weight) {
    bitvector c(blocksize);
    std::vector<size_t> e = random_subset<size_t>(blocksize, error_weight, eng);
    for(const auto &i : e) c.flip(i);
    return c;
}

//not performance optimized but good enough
bitmatrix submatrix(const bitmatrix &B, const bitvector &sub) {
    size_t n = sub.size();
    if(B.num_rows() != B.num_cols() || n != B.num_rows()) {
        notify("Error: dimension mismatch in submatrix.");
        return B;
    }
    bitmatrix Bprime;
    for(size_t i = 0; i < n; i++) if(sub.get(i)) Bprime.append_below(B.row(i));
    bitmatrix returnval;
    for(size_t i = 0; i < n; i++) if(sub.get(i)) returnval.append_right(Bprime.col(i));
    return returnval;
}

bitmatrix direct_sum(const bitmatrix &A, const bitmatrix &B) {
    bitmatrix upper_right_block(A.num_rows(), B.num_cols()); //all zeros
    bitmatrix lower_left_block(B.num_rows(), A.num_cols());  //all zeros
    bitmatrix returnval = A;
    returnval.append_right(upper_right_block);
    bitmatrix bottom_half = lower_left_block;
    bottom_half.append_right(B);
    returnval.append_below(bottom_half);
    return returnval;
}

bitmatrix direct_sum(const std::vector<bitmatrix> &blocks) {
    size_t num_blocks = blocks.size();
    if(num_blocks == 0) {
        notify("Error: empty call to direct_sum");
        bitmatrix returnval(0,0);
        return returnval;
    }
    if(num_blocks == 1) return blocks[0];
    bitmatrix returnval = direct_sum(blocks[0], blocks[1]);
    for(size_t i = 2; i < num_blocks; i++) returnval = direct_sum(returnval, blocks[i]);
    return returnval;
}

bitmatrix block(const bitmatrix &B, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    if(row_start >= row_end || row_end > B.num_rows() || col_start >= col_end || col_end > B.num_cols()) {
        notify("Error: invalid ranges in block.");
        bitmatrix returnval(0,0);
        return returnval;
    }
    bitmatrix C = B.row_range(row_start, row_end);
    return C.col_range(col_start, col_end);
}