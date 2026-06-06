#include <iostream>
#include "bitmatrix.hpp"
#include "xorsat.hpp"
#include "ldpc.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: ldgmsat instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    if(!load_instance(argv[1], I)) return 0;
    bitmatrix H, M, ID;
    H = parity_check_matrix(I);
    size_t m = H.num_cols();
    size_t n = H.num_rows();
    M = H.transpose();
    ID.identity(m);
    M.append_right(ID);
    bp dec(M);
    std::vector<double> p_flip(n+m);
    for(size_t i = 0; i < n; i++) p_flip[i] = 0.5;
    bitvector decoded, x;
    bitvector best_x(n);
    bitvector targ(n); //the first n entries don't matter and can be all zero
    targ.append(target(I));
    double p, val;
    double best_val = 0;
    std::cout << "H:" << std::endl;
    std::cout << H << std::endl;
    std::cout << "M:" << std::endl;
    std::cout << M << std::endl;
    std::cout << "target:" << std::endl;
    std::cout << target(I) << std::endl;
    std::cout << "targ:" << std::endl;
    std::cout << targ << std::endl;
    for(int p_index = 1; p_index <= 10; p_index++) {
        p = 0.05*p_index;
        for(size_t i = n; i < n+m; i++) p_flip[i] = p;
        std::cout << vec_string(p_flip, " ") << std::endl;
        decoded = dec.bp_decode(targ, p_flip);
        std::cout << "decoded: " << decoded << std::endl;
        x = decoded.range(0,n);
        val = evaluate(I, x);
        if(p_index == 1 || val < best_val) {
            best_val = val;
            best_x = x;
        }
        std::cout << "p: " << p << " val: " << val << std::endl;
    }
    std::cout << "best val: " << best_val << std::endl;
    std::cout << "best x: " << best_x << std::endl;
    return 0;
}
