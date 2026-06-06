#ifndef PANNEAL_HPP
#define PANNEAL_HPP

#include <random>
#include <iostream>
#include <cmath>
#include "utils.hpp"
#include "listsat.hpp"

//plain vanilla single-threaded simulated annealing with linearly increasing inverse temperature
template<typename T>
void anneal(T &I, std::mt19937_64 &eng) {
    std::uniform_int_distribution<uint64_t> distr;
    std::uniform_int_distribution<uint16_t> rand_diff(1,I.x.p()-1);
    I.x.random(eng); //initialize randomly
    int val = evaluate(I, I.x);
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
    pvector best_x = I.x;
    int iterations = 10000;
    double beta = 0;
    double delta_beta = 20.0/iterations; //just set maxbeta = 20
    uint16_t new_val;
    for(t = 0; t < iterations; t++) { //increment timestep
        beta += delta_beta;
        for(i = 0; i < I.x.size(); i++) { //sweep through variable index
            accept = false;
            new_val = (I.x.get(i) + rand_diff(eng))%I.x.p(); //ensures the new val will not equal the old val
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
        }
        if(t%10 == 0) {
            std::cout << fwi(t+1) << fwf(beta) << fwf(val) << fwf((double)accepted/(double)(accepted+rejected)) << std::endl;
            accepted = 0;
            rejected = 0;
        }
    }
    int final_val = evaluate(I,I.x);
    if(final_val != val) {
        std::cout << "ERROR: faulty diffential evaluation!" << std::endl;
    }
    std::cout << "final x:" << std::endl;
    std::cout << I.x << std::endl;
    std::cout << "best x:" << std::endl;
    std::cout << best_x << std::endl;
    std::cout << fws("n") << fws("m") << fws("sweeps") << fws("val") << fws("best_val") << std::endl;
    std::cout << fwi(I.M.num_rows()) << fwi(I.M.num_cols()) << fwi(t) << fwi(evaluate(I,I.x)) << fwi(best_val) << std::endl;
    std::cout << I.M.num_cols() - best_val << " clauses satisfied out of " << I.M.num_cols() << std::endl;
}

#endif
