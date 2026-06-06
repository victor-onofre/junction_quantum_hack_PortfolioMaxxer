#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include "utils.hpp"
#include "advrand.hpp"

#define TRIALS 1000

int main(int argc, char *argv[]) {
    sparse_xorsat I;
    if(argc != 2) {
        std::cout << "Usage: advrand instance.tsv" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    if(!I.load(argv[1])) return 0;
    std::vector<int> x(I.num_vars);
    std::vector<int> bestx(I.num_vars);
    std::vector<double>grad(I.num_vars);
    size_t numrand;
    double avg;
    double avg_ascensions;
    double val, bestval;
    bestval = -100000;
    for(numrand = 0; numrand <= x.size(); numrand++) { //exhaustive hyperparameter sweep (scan over all values of numrand)
        avg = 0.0;
        avg_ascensions = 0.0;
        for(size_t trial = 0; trial < TRIALS; trial++) {
            partial_random(x, numrand, eng);
            partial_gradient(I, x, grad);
            avg_ascensions += ascend(x, grad, eng)/(double)TRIALS;
            val = I.value(x);
            avg += val/(double)TRIALS;
            if(val > bestval) {
                bestval = val;
                bestx = x;
            }
            //std::cout << "value before perturbation: " << I.value(x) << std::endl;
            //perturb(x, 0.01, eng);
            //std::cout << "value after perturbation: " << I.value(x) << std::endl;
        }
        std::cout << "numrand: " << numrand << "/" << I.num_vars << " avg: " << avg << " avg ascensions: " << avg_ascensions << std::endl;
    }
    std::cout << "bestval: " << bestval << std::endl;
    std::cout << (int)std::round(bestval + 0.5*(double)I.num_cons) << " clauses satisfied out of " << I.num_cons << std::endl;
    std::string infile = argv[1];
    std::string outfile = infile + ".sol";
    std::ofstream of(outfile);
    for(int xi : bestx) {
        if(xi == -1) of << "1";
        if(xi == 1) of << "0";
        if(xi != -1 && xi != 1) notify("Error: invalid value in bestx");
    }
    of.close();
    std::cout << "solution saved to " << outfile << std::endl;
    return 0;
}

//testing gradient
/*int main(int argc, char *argv[]) {
    sparse_xorsat I;
    if(argc != 2) {
        std::cout << "Usage: advrand instance.tsv" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = 3657347230;//rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    if(!I.load(argv[1])) return 0;
    std::vector<int> x(I.num_vars);
    size_t numrand = (size_t)std::round(0.75*(double)x.size());
    partial_random(x, numrand, eng);
    std::vector<double>grad(x.size());
    partial_gradient(I, x, grad);
    std::cout << "Instance:\n" << I.to_string() << std::endl;
    std::cout << "Partial assignment:\n" << vec_string(x, " ") << std::endl;
    std::cout << "Partial Gradient:\n" << vec_string(grad, " ") << std::endl;
    return 0;
}*/

//testing evaluation
/*int main(int argc, char *argv[]) {
    sparse_xorsat I;
    double val;
    if(argc != 3) {
        std::cout << "Usage: advrand instance.tsv solution.sol" << std::endl;
        return 0;
    }
    if(!I.load(argv[1])) return 0;
    std::vector<int> x = load_sol(argv[2]);
    if(x.size() == 0) return 0;
    val = I.value(x);
    std::cout << val << std::endl;
    return 0;
}*/

//testing file IO
/*int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: advrand instance.tsv" << std::endl;
        return 0;
    }
    sparse_xorsat I;
    std::string input_filename = argv[1];
    if(!I.load(input_filename)) return 0;
    std::string output_filename = "out_" + input_filename;
    I.save(output_filename);
    return 0;
}*/