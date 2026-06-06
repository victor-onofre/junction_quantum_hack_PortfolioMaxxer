#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include "utils.hpp"
#include <cmath>

#define PRINT_HIST false

size_t maxD(const instance &I) {
    size_t D, maxD;
    maxD = 0;
    for(const unweighted_instance &UI : I.U)  {
        for(const bitvector &var : UI.v) {
            D = var.count();
            if(D > maxD) maxD = D;
        }
    }
    return maxD;
}

std::vector<int> histogram(const instance &I, int maxD) {
    std::vector<int> returnval(maxD+1);
    bitvector overlap(I.num_cons);
    int energy;
    for(const unweighted_instance &UI : I.U)  {
        for(const bitvector &var : UI.v) {
            overlap = var & UI.s;
            energy = overlap.count();
            returnval[energy] += 1;
        }
    }
    return returnval;
}

double delta(const instance &I, size_t var_index) {
    if(var_index > I.num_vars) {
        notify("Error: var index out of bounds in delta");
        return 1.0;
    }
    double delta_squared = 0.0;
    size_t weight_index, var_degree;
    for(weight_index = 0; weight_index < I.w.size(); weight_index++) {
        var_degree = I.U[weight_index].v[var_index].count();
        delta_squared += var_degree*I.w[weight_index]*I.w[weight_index];
    }
    return sqrt(delta_squared);
}

//If we imagine that flipping variable i has a 50% chance of improving vs. worsening each
//clause that its in then delta[i] is the RMS change in cost value, modulo stray factors
//of sqrt(2) or whatever that we don't care about because we multiply by an empirical fudge
//factor at the end anyway.
double maxbeta(const instance &I) {
    double delta_i, min_delta;
    min_delta = delta(I, 0);
    for(size_t var_index = 1; var_index < I.num_vars; var_index++) {
        delta_i = delta(I, var_index);
        if(delta_i < min_delta) min_delta = delta_i;
    }
    return 6.0/min_delta;
}

//plain vanilla single-threaded simulated annealing with linearly increasing inverse temperature
void anneal(instance &I, std::mt19937_64 &eng) {
    size_t D = maxD(I);
    std::vector<int> hist(D+1);
    std::vector<int> sweephist(D+1);
    std::uniform_int_distribution<uint64_t> distr;
    bitvector x(I.num_vars);
    x.random(eng); //initialize randomly
    double val = evaluate(I, x);
    int t;
    unsigned int i;
    double delta;
    double p_accept;
    uint64_t threshold;
    int accepted = 0;
    int rejected = 0;
    bool accept;
    std::cout << fws("sweep") << fws("beta") << fws("val") << fws("frac_acc") << std::endl;
    double best_val = val;
    bitvector best_x = x;
    int iterations = 5000;
    double beta = 0;
    double delta_beta = maxbeta(I)/iterations;
    for(t = 0; t < iterations; t++) { //increment timestep
        beta += delta_beta;
        if(t%10==0 && PRINT_HIST) std::fill(sweephist.begin(), sweephist.end(), 0);
        for(i = 0; i < I.num_vars; i++) { //sweep through variable index
            accept = false;
            delta = diff(I, i);
            if(delta <= 0) accept = true;
            if(!accept) {
                p_accept = exp(-1.0*beta*delta);
                threshold = (uint64_t)(p_accept*(double)UINT64_MAX);
                if(distr(eng) < threshold) accept = true;
            }
            if(accept) {
                val += delta;
                flip(I, i);
                x.flip(i);
                accepted++;
            }
            else rejected++;
            if(val < best_val) {
                best_val = val;
                best_x = x;
            }
        }
        if(t%10 == 0) {
            if(PRINT_HIST) {
                hist = histogram(I, D);
                for(size_t i = 0; i < D+1; i++) sweephist[i] += hist[i];
            }
            std::cout << fwi(t+1) << fwf(beta) << fwf(val) << fwf((double)accepted/(double)(accepted+rejected)) << std::endl;
            if(PRINT_HIST) std::cout << "H " <<  vec_string(sweephist, " ") << std::endl;
            accepted = 0;
            rejected = 0;
        }
    }
    double final_val = evaluate(I,x);
    if(fabs(final_val - val) > 0.001) {
        std::cout << "ERROR: faulty diffential evaluation!" << std::endl;
    }
    std::cout << "final x:" << std::endl;
    std::cout << x << std::endl;
    std::cout << "best x:" << std::endl;
    std::cout << best_x << std::endl;
    std::cout << fws("n") << fws("m") << fws("sweeps") << fws("val") << fws("best_val") << std::endl;
    std::cout << fwi(I.num_vars) << fwi(I.num_cons) << fwi(t) << fwf(evaluate(I,x)) << fwf(best_val) << std::endl;
    if (I.w.size() == 1) {
        int satisfied = (double(I.num_cons) - double(best_val)/double(I.w.at(0)))/2.0;
        std::cout << satisfied << " clauses satisfied out of " << I.num_cons<<std::endl;
    } else {
        std::cout << "Unknown number of satisfied clauses due to different weights." << std::endl;
    }
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: sawxor instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::mt19937_64 eng(seed);
    std::string filename = argv[1];
    if(!load_instance(filename, I)) return 0;
    //cout << to_string(I) << endl;
    anneal(I, eng);
    return 0;
}
