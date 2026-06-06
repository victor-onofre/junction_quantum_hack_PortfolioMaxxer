#ifndef LDPC_HPP
#define LDPC_HPP

#include <vector>
#include "bitmatrix.hpp"
#include "xorsat.hpp"

//convert from unweighted xorsat instance to parity check matrix
bitmatrix parity_check_matrix(const unweighted_instance &I);

//convert from xorsat instance to parity check matrix
bitmatrix parity_check_matrix(const instance &I);

//convert from parity check matrix to xorsat instance
instance PCM_to_instance(const bitmatrix &H);

bitmatrix reverse_cols(const bitmatrix &M);

bool is_pc_standard_form(const bitmatrix &H);

//R H P = NH, R is general linear, P is permutation, NH is normal form.
//Will drop all-zero rows from bottom of NH if H is not full rank.
void pc_standard_form(const bitmatrix &H, bitmatrix &NH, bitmatrix &P);

void generator_matrix(const bitmatrix &H, bitmatrix &G);

typedef struct bp_node bp_node;

struct bp_node {
    double check_to_bit;
    double bit_to_check;
    size_t i;
    size_t j;
    bp_node *up;
    bp_node *down;
    bp_node *left;
    bp_node *right;
};

//Not thread safe. Make separate object for each thread
class bp {
    public:
        bp();
        bp(const bitmatrix &parity_check_matrix);
        void init(const bitmatrix &parity_check_matrix);
        bitvector bp_decode(const bitvector &corrupted_codeword, double p_flip);
        bitvector bp_decode(const bitvector &corrupted_codeword, std::vector <double> p_flip, int maxiter=-1);
        void dump();
        int iterations(); //returns the number of iterations used in the most recent decode
        bitmatrix H;
        bitmatrix HT; //H transpose
    private:
        bool ready;
        int iters;
        std::vector <bp_node> e;
        std::vector <bp_node *>row_start;
        std::vector <bp_node *>row_end;
        std::vector <bp_node *>col_start;
        std::vector <bp_node *>col_end;
};

bitvector rand_word(const bitmatrix &G, std::mt19937_64 &eng);

bitvector apply_rand_err(const bitvector &uncorrupted, std::mt19937_64 &eng, size_t k);

//parity check and generator matrices from the Gallager ensemble
void gallager(size_t k, size_t D, size_t bsize, bitmatrix &H, bitmatrix &G, std::mt19937_64 &eng);

//generate parity check matrix uniformly at random and calculate generator matrix too
void dense(size_t n, size_t m, bitmatrix &H, bitmatrix &G, std::mt19937_64 &eng);

#endif
