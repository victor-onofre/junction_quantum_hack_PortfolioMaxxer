#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include "utils.hpp"
#include "maxlinsat.hpp"

//For simulated annealing we don't need the modulus to be prime and we don't need to compute inverses mod p.
//So there is no need to limit the modulus to the range treated in modpinverse.hpp, which currently tops out at 512.

//plain vanilla single-threaded simulated annealing with linearly increasing inverse temperature
void anneal(modinstance &I, std::mt19937_64 &eng) {
    std::uniform_int_distribution<uint64_t> distr;
    std::uniform_int_distribution<uint64_t> rand_diff(1,I.modulus-1);
    randomize_pvec(I.x, I.modulus, eng);
    int val = evaluate(I);
    std::cout << "initial value: " << val << std::endl;
    int t;
    unsigned int i;
    int delta;
    double p_accept;
    uint64_t threshold;
    int accepted = 0;
    int rejected = 0;
    bool accept;
    std::cout << fws("sweep") << fws("beta") << fws("val") << fws("frac_acc") << std::endl;
    int best_val = val;
    std::vector<uint64_t> best_x = I.x;
    int iterations = 10000;
    double beta = 0;
    double maxbeta = 5.0*pow((double)I.num_cons,-0.43); //This is an empirical power law. Theory would suggest a power of -0.5.
    double delta_beta = maxbeta/iterations;
    uint64_t new_val;
    for(t = 0; t < iterations; t++) { //increment timestep
        beta += delta_beta;
        for(i = 0; i < I.x.size(); i++) { //sweep through variable index
            accept = false;
            new_val = (I.x.at(i) + rand_diff(eng))%I.modulus; //ensures the new val will not equal the old val
            delta = diff(I, i, new_val);
            if(delta <= 0) accept = true;
            if(!accept) {
                p_accept = exp(-1.0*beta*(double)delta);
                threshold = (uint64_t)(p_accept*(double)UINT64_MAX);
                if(distr(eng) < threshold) accept = true;
            }
            if(accept) {
                val += delta;
                assign(I, i, new_val);
                accepted++;
            }
            else rejected++;
            if(val < best_val) {
                best_val = val;
                best_x = I.x;
            }
            //This is slow and just for testing
            //int trueval = evaluate_at(I, I.x);
            //if(trueval != val) std::cout << "Failure: val = " << val << " trueval = " << trueval << std::endl;
        }
        if(t%10 == 0) {
            std::cout << fwi(t+1) << fwf(beta) << fwf(val) << fwf((double)accepted/(double)(accepted+rejected)) << std::endl;
            accepted = 0;
            rejected = 0;
        }
    }
    int final_val = evaluate(I);
    if(final_val != val) std::cout << "ERROR: faulty diffential evaluation!" << std::endl;
    std::cout << "final x:" << std::endl;
    std::cout << pvec_string(I.x) << std::endl;
    std::cout << "best x:" << std::endl;
    std::cout << pvec_string(best_x) << std::endl;
    std::cout << fws("n") << fws("m") << fws("sweeps") << fws("val") << fws("best_val") << std::endl;
    std::cout << fwi(I.num_vars) << fwi(I.num_cons) << fwi(t) << fwi(final_val) << fwi(best_val) << std::endl;
    std::cout << I.num_cons - best_val << " clauses satisfied out of " << I.num_cons << std::endl;
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: bigpanneal instance.csv" << std::endl;
        return 0;
    }
    modinstance I;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    std::string filename = argv[1];
    if(!load_modinstance(filename, I)) return 0;
    anneal(I, eng);
    return 0;
}

//This is a test. For modulus below 512, the results should match evalist.
//It passed on data/ps1.csv data/ps1.psol, data/ps2.csv data/ps2.psol, and data/ps3.csv data/psol
/*int main(int argc, char *argv[]) {
    modinstance I;
    std::vector<uint64_t> x;
    double val;
    if(argc != 3) {
        std::cout << "Usage: bigpanneal instance.csv solution.psol" << std::endl;
        return 0;
    }
    if(!load_modinstance(argv[1], I)) return 0;
    std::cout << "Loaded instance " << argv[1] << " with " << I.num_vars << " variables and " << I.num_cons << " clauses." << std::endl;
    //std::cout << modinstance_string(I);
    x = load_solution(argv[2], I.modulus);
    //for(const uint64_t & val : x) std::cout << val << ','; std::cout << std::endl;
    if(x.size() == 0) return 0;
    std::cout << "Loaded solution " << argv[2] << " with " << x.size() << " variables." << std::endl << std::endl;
    //val is the number of unsatisfied clauses
    val = evaluate(I, x);
    std::cout << "number of unsatisfied clauses: " << val << std::endl;
    std::cout << "number of satisfied clauses: " << I.num_cons - val << std::endl;
    std::cout << "(SAT - UNSAT): " << I.num_cons - 2*val << std::endl;
    std::cout << "(1/2)*(SAT - UNSAT): " << 0.5*(I.num_cons - 2.0*val) << std::endl;
    return 0;
}*/

//This is a test of differential evaluation.
//It passed.
/*int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: bigpanneal instance.csv" << std::endl;
        return 0;
    }
    modinstance I;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    std::string filename = argv[1];
    if(!load_modinstance(filename, I)) return 0;
    //std::cout << modinstance_string(I);
    std::vector<uint64_t> x(I.num_vars);
    randomize_pvec(x, I.modulus, eng);
    I.x = x;
    int violated = evaluate(I);
    std::bernoulli_distribution coinflip(0.5);
    std::uniform_int_distribution<uint64_t> diceroll(0, I.modulus);
    for(size_t i = 0; i < I.num_vars; i++) {
        uint64_t newval = diceroll(eng);
        int delta = diff(I, i, newval);
        //int before = evaluate_at(I,I.x);
        //std::vector<uint64_t> newx = I.x;
        //newx[i] = newval;
        //int after = evaluate_at(I,newx);
        //int slowdelta = after - before;
        //std::cout << delta << " vs " << slowdelta << std::endl;
        if(coinflip(eng)) {
            assign(I, i, newval);
            violated += delta;
        }
    }
    std::cout << pvec_string(I.x) << std::endl;
    std::cout << violated << " constraints violated according to differential evaluation." << std::endl;
    std::cout << evaluate_at(I, I.x) << " constraints violated according to direct evaluation." << std::endl;
    return 0;
}*/
