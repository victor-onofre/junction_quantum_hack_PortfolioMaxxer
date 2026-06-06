#include <iostream>
#include <vector>
#include <list>
#include <cmath>
#include "utils.hpp"
#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include "ldpc.hpp"

bitmatrix parity_check_matrix(const unweighted_instance &I) {
    bitmatrix H(I.v.size(), I.c.size());
    for(size_t var = 0; var < I.v.size(); var++) {
        for(size_t con = 0; con < I.c.size(); con++) {
            if(I.v[var].get(con)) H.set(var, con, 1);
        }
    }
    return H;
}

bitmatrix parity_check_matrix(const instance &I) {
    bitmatrix H = parity_check_matrix(I.U[0]);
    for(size_t weight_index = 1; weight_index < I.U.size(); weight_index++) {
        H.append_right(parity_check_matrix(I.U[weight_index]));
    }
    return H;
}

bitmatrix reverse_cols(const bitmatrix &M) {
    bitmatrix returnval;
    returnval = M.col(M.num_cols() - 1);
    for(size_t i = M.num_cols()-2; i >= 0; i++) returnval.append_right(M.col(i));
    return returnval;
}

bool is_pc_standard_form(const bitmatrix &H) {
    if(H.num_cols() < H.num_rows()) return false;
    bitmatrix R = H.col_range(H.num_cols()-H.num_rows(), H.num_cols());
    bitmatrix I;
    I.identity(H.num_rows());
    if(R == I) return true;
    return false;
}

//R H P = NH, R is general linear, P is permutation, NH is normal form.
//Will drop all-zero rows from bottom of NH if H is not full rank.
void pc_standard_form(const bitmatrix &H, bitmatrix &NH, bitmatrix &P) {
    if(is_pc_standard_form(H)) {
        NH = H;
        P.identity(H.num_cols());
        return;
    }
    bitmatrix H1 = H.reduced_row_echelon_form();
    size_t nonzeros = 0;
    for(size_t row_index = 0; row_index < H1.num_rows(); row_index++) {
        bitvector rowvec = H1.row_vec(row_index);
        if(rowvec.count() > 0) nonzeros++;
    }
    if(nonzeros < H1.num_rows()) {
        notify("Note: parity check matrix of " + std::to_string(H1.num_rows()) + " rows had rank " + std::to_string(nonzeros));
        bitmatrix tmp = H1.row_range(0, nonzeros);
        H1 = tmp;
    }
    std::list<size_t> perm;
    std::vector<bool> used(H1.num_cols(), false);
    size_t row = 0;
    for(size_t col = 0; col < H1.num_cols() && row < H1.num_rows(); col++) {
        if(H1.get(row, col) == 1) {
            perm.push_back(col);
            used[col] = true;
            row++;
        }
    }
    for(size_t col = 0; col < H1.num_cols(); col++) if(!used[col]) perm.push_front(col);
    P.zeros(H1.num_cols(), H1.num_cols());
    size_t col = 0;
    for(const size_t &p : perm) P.set(p, col++, 1);
    NH = H1 * P;
}

void generator_matrix(const bitmatrix &H, bitmatrix &G) {
    bitmatrix P;
    bitmatrix NH;
    pc_standard_form(H, NH, P);
    bitmatrix L = NH.col_range(0,NH.num_cols() - NH.num_rows());
    bitmatrix id;
    id.identity(NH.num_cols()-NH.num_rows());
    G = hstack(id, L.transpose()) * P.transpose();
}

bp::bp() {
    ready = false;
}

bp::bp(const bitmatrix &parity_check_matrix) {
    init(parity_check_matrix);
}

void bp::init(const bitmatrix &parity_check_matrix) {
    H = parity_check_matrix;
    row_end.resize(H.num_rows());
    row_start.resize(H.num_rows());
    col_end.resize(H.num_cols());
    col_start.resize(H.num_cols());
    std::vector<std::vector<size_t> > row_edge(H.num_rows());
    std::vector<std::vector<size_t> > col_edge(H.num_cols());
    bp_node edge;
    for(size_t i = 0; i < H.num_rows(); i++) {
        for(size_t j = 0; j < H.num_cols(); j++) {
            if(H.get(i,j)) {
                edge.left = NULL;
                edge.right = NULL;
                edge.down = NULL;
                edge.up = NULL;
                edge.i = i;
                edge.j = j;
                e.push_back(edge);
                row_edge[i].push_back(e.size()-1);
                col_edge[j].push_back(e.size()-1);
            }
        }
    }
    //fill in the left and right pointers
    for(size_t row_index = 0; row_index < H.num_rows(); row_index++) {
        for(size_t entry_index = 0; entry_index < row_edge[row_index].size(); entry_index++) {
            size_t ei = row_edge[row_index][entry_index];
            if(entry_index > 0) e[ei].left = &(e[row_edge[row_index][entry_index-1]]);
            if(entry_index < row_edge[row_index].size()-1) e[ei].right = &(e[row_edge[row_index][entry_index+1]]);
        }
    }
    //fill in the up and down pointers
    for(size_t col_index = 0; col_index < H.num_cols(); col_index++) {
        for(size_t entry_index = 0; entry_index < col_edge[col_index].size(); entry_index++) {
            size_t ei = col_edge[col_index][entry_index];
            if(entry_index > 0) e[ei].up = &(e[col_edge[col_index][entry_index-1]]);
            if(entry_index < col_edge[col_index].size()-1) e[ei].down = &(e[col_edge[col_index][entry_index+1]]);
        }
    }
    for(size_t edge_index = 0; edge_index < e.size(); edge_index++) {
        if(e[edge_index].up == NULL) col_start[e[edge_index].j] = &(e[edge_index]);
        if(e[edge_index].down == NULL) col_end[e[edge_index].j] = &(e[edge_index]);
        if(e[edge_index].left == NULL) row_start[e[edge_index].i] = &(e[edge_index]);
        if(e[edge_index].right == NULL) row_end[e[edge_index].i] = &(e[edge_index]);
    }
    HT = H.transpose();
    ready = true;
}

/* 
    This is basically a C++ port of Joschka Roffe's python code from:
    https://github.com/quantumgizmos/ldpc/blob/main/src/ldpc/bp_decoder.pyx
    A maxiter of -1 means set it automatically.
*/
bitvector bp::bp_decode(const bitvector &corrupted_codeword, std::vector <double> p_flip, int maxiter) {
    if(!ready) {
        notify("BP not ready");
        bitvector empty;
        return empty;
    }
    if (maxiter == -1) {
        double expecterr = 0.0;
        for(auto &p : p_flip) {
            expecterr += p;
            if(p > 1.0) { notify("p_flip must be at most 1.0"); return corrupted_codeword; }
            if(p == 1.0) p = 1.0 - 1.0/(double)H.num_cols();
            if(p < 0.0) { notify("p_flip must be nonnegative"); return corrupted_codeword; }
            if(p == 0.0) p = 1.0/(double)H.num_cols();
        } //only local copy is modified
        maxiter = (size_t)round(expecterr);
        if(maxiter < 10) maxiter = 10;
        if(maxiter > 5000) maxiter = 5000;
    }
    size_t i,j;
    double temp;
    bp_node *ep;
    bitvector syndrome = H * corrupted_codeword;
    bitvector bp_decoding(H.num_cols());
    bitvector bp_syndrome(H.num_rows());
    bitvector prior_bp_decoding(H.num_cols());
    //initialize:
    for(auto &er : e) er.bit_to_check = p_flip[er.j]/(1.0 - p_flip[er.j]);
    for(int iter = 0; iter < maxiter; iter++) {
        iters = iter;
        //update check-to-bit messages using product-sum rule
        for(i = 0; i < H.num_rows(); i++) {
            temp = 1.0;
            if(syndrome.get(i)) temp = -1.0;
            for(ep = row_start[i]; ep != NULL; ep = ep->right) {
                ep->check_to_bit = temp;
                temp *= 2.0/(1.0 + ep->bit_to_check) - 1.0;
            }
            temp = 1.0;
            for(ep = row_end[i]; ep != NULL; ep = ep->left) {
                ep->check_to_bit *= temp;
                ep->check_to_bit = (1.0 - ep->check_to_bit)/(1.0 + ep->check_to_bit);
                temp *= 2.0/(1.0+ep->bit_to_check) - 1.0;
            }
        }
        //update bit-to-check messages
        for(j = 0; j < H.num_cols(); j++) {
            temp = p_flip[j]/(1.0 - p_flip[j]);
            for(ep = col_start[j]; ep != NULL; ep = ep->down) {
                ep->bit_to_check = temp;
                temp *= ep->check_to_bit;
                if(std::isnan(temp)) { notify("isnan temp(1)"); temp = 1.0; }
            }
            if(temp >= 1.0) bp_decoding.set(j,1);
            else bp_decoding.set(j,0);
            temp = 1.0;
            for(ep = col_end[j]; ep != NULL; ep = ep->up) {
                ep->bit_to_check *= temp;
                temp *= ep->check_to_bit;
                if(std::isnan(temp)) { notify("isnan temp(2)"); temp = 1.0; }
            }
        }
        bp_syndrome += (bp_decoding + prior_bp_decoding) * HT; //differential evaluation is usually sparse and hence cheaper
        if(bp_syndrome == syndrome) {
            return corrupted_codeword + bp_decoding;
        }
        prior_bp_decoding = bp_decoding;
    }
    return corrupted_codeword + bp_decoding;
}

bitvector bp::bp_decode(const bitvector &corrupted_codeword, double p_flip) {
    if(!ready) {
        notify("BP not ready");
        bitvector empty;
        return empty;
    }
    if(p_flip > 1.0) { notify("Error: p_flip must be at most 1.0"); return corrupted_codeword; }
    if(p_flip < 0.0) { notify("Error: p_flip must be nonnegative"); return corrupted_codeword; }
    std::vector <double>pf(H.num_cols(), p_flip);
    return bp_decode(corrupted_codeword, pf);
}

//convert from parity check matrix to xorsat instance
instance PCM_to_instance(const bitmatrix &H) {
    instance I;
    unweighted_instance UI;
    I.num_cons = H.num_cols();
    I.num_vars = H.num_rows();
    //0.5 is a convenient normalization because when one constraint goes from unsatisfied to satisfied
    //the cost function will decrease by one.
    I.w.push_back(0.5);
    for(size_t row_index = 0; row_index < H.num_rows(); row_index++) UI.v.push_back(H.row_vec(row_index));
    for(size_t col_index = 0; col_index < H.num_cols(); col_index++) UI.c.push_back(H.col_vec(col_index));
    UI.b.zeros(H.num_cols()); //parity check matrix does not determine b.
    UI.s.zeros(H.num_cols()); //This is just to allocate the memory.
    I.U.push_back(UI);
    return I;
}

void bp::dump() {
    if(!ready) {
        notify("BP not ready");
        return;
    }
    for(auto const & ep : e) {
        std::cout << ep.i << " " << ep.j << ":" << std::endl;
        if(ep.up == NULL) std::cout << "up: NULL" << std::endl;
        else std::cout << "up: " << ep.up->i << " " << ep.up->j << std::endl;
        if(ep.down == NULL) std::cout << "down: NULL" << std::endl;
        else std::cout << "down: " << ep.down->i << " " << ep.down->j << std::endl;
        if(ep.left == NULL) std::cout << "left: NULL" << std::endl;
        else std::cout << "left: " << ep.left->i << " " << ep.left->j << std::endl;
        if(ep.right == NULL) std::cout << "right: NULL" << std::endl;
        else std::cout << "right: " << ep.right->i << " " << ep.right->j << std::endl;
    }
}

bitvector rand_word(const bitmatrix &G, std::mt19937_64 &eng) {
    bitvector input(G.num_rows());
    input.random(eng);
    return input * G;
}

bitvector apply_rand_err(const bitvector &uncorrupted, std::mt19937_64 &eng, size_t k) {
    std::vector<size_t> e = random_subset<size_t>(uncorrupted.size(), k, eng);
    bitvector corrupted = uncorrupted;
    for(const auto &i : e) corrupted.flip(i);
    return corrupted;
}

int bp::iterations() {
    if(!ready) {
        notify("BP not ready.");
        return 0;
    }
    return iters;
}

//parity check and generator matrices from the Gallager ensemble
void gallager(size_t k, size_t D, size_t bsize, bitmatrix &H, bitmatrix &G, std::mt19937_64 &eng) {
    size_t m = D*bsize;
    size_t n = k*bsize;
    bitmatrix B(n,m);
    std::vector<int> perm(m);
    for(size_t i = 0; i < m; i++) perm[i] = i;
    for(size_t a = 0; a < k; a++) {
        std::shuffle(perm.begin(), perm.end(), eng);
        for(size_t b = 0; b < bsize; b++) {
            size_t row_index = a*bsize+b;
            for(size_t entry_index = 0; entry_index < D; entry_index++) {
                B.set(row_index, perm[bsize*entry_index + row_index%bsize], 1);
            }
        }
    }
    H = B;
    generator_matrix(H, G);
}

//generate parity check matrix uniformly at random and calculate generator matrix too
void dense(size_t n, size_t m, bitmatrix &H, bitmatrix &G, std::mt19937_64 &eng) {
    bitmatrix B(n,m);
    std::bernoulli_distribution coin(0.5);
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; j++) {
            B.set(i,j,coin(eng));
        }
    }
    H = B;
    generator_matrix(H, G);
}
