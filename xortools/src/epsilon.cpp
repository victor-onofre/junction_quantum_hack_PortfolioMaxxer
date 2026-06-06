#include "xorsat.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>

int main(int argc, char *argv[]) {
    if(argc != 3) {
        std::cout << "Usage: epsilon l instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    unsigned int l = std::stoi(argv[1]);
    int success = load_instance(argv[2], I);
    if(!success) return 0;
    if(2*l > I.num_cons) {
        std::cout << "l = " << l << " is larger than half the number of constraints (" << I.num_cons << ") which is not plausible." << std::endl;
        return 0;
    }
    std::vector <double> M;
    std::vector <int> s;
    std::cout << "Magnitude stats:" << std::endl;
    magnitudes(I, M, s);
    std::cout << vec_string(M,"\t") << std::endl;
    std::cout << vec_string(s, "\t") << std::endl;
    if(M.size() == 1) {
        std::cout << "M*sqrt(2ml) = " << M[0]*sqrt((double)(2*I.num_cons*l)) << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
    pstats st;
    stats(M,s,l,st);
    print_pstats(st);
    return 0;
}
