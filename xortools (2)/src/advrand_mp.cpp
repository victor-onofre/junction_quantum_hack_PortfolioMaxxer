#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <thread>
#include "utils.hpp"
#include "advrand.hpp"

#define TRIALS 1000

typedef struct {
    uint64_t seed;
    size_t num_trials;
    size_t num_rand;
    int *total_trials;
    int *total_ascensions;
    double *total_val;
    sparse_xorsat *I;
    double *best_val;
    std::vector<int> *best_x;
    std::mutex *mtx;
}trialdata;

void advrand_trial(trialdata t) {
    //std::string s = "seed: " + std::to_string(t.seed) + " num_trials: " + std::to_string(t.num_trials) + " num_rand: " + std::to_string(t.num_rand) + '\n';
    //std::cout << s << std::flush;
    std::mt19937_64 eng(t.seed);
    std::vector<int> x(t.I->num_vars);
    std::vector<double>grad(t.I->num_vars);
    std::vector<int> local_best_x(t.I->num_vars);
    double local_best_val = 0.0;
    double local_total_val = 0.0;
    double val = 0.0;
    int ascensions = 0;
    int local_total_ascensions = 0;
    for(size_t trial = 0; trial < t.num_trials; trial++) {
        partial_random(x, t.num_rand, eng);
        partial_gradient(*(t.I), x, grad);
        ascensions = ascend(x, grad, eng);
        local_total_ascensions += ascensions;
        val = t.I->value(x);
        local_total_val += val;
        if(trial == 0 || val > local_best_val) {
            local_best_val = val;
            local_best_x = x;
        }
    }
    t.mtx->lock();//------------------------------------------------------------------------
    *(t.total_trials) += t.num_trials;
    *(t.total_val) += local_total_val;
    *(t.total_ascensions) += local_total_ascensions;
    if(local_best_val > *(t.best_val)) {
        *(t.best_val) = local_best_val;
        *(t.best_x) = local_best_x;
    }
    t.mtx->unlock();//----------------------------------------------------------------------
}

int main(int argc, char *argv[]) {
    sparse_xorsat I;
    if(argc != 2) {
        std::cout << "Usage: advrand_mp instance.tsv" << std::endl;
        return 0;
    }
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    if(!I.load(argv[1])) return 0;
    size_t numrand;
    double avg;
    double avg_ascensions;
    uint64_t num_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " threads." << std::endl;
    trialdata t;
    int total_trials = 0;
    t.total_trials = &total_trials;
    int total_ascensions = 0;
    t.total_ascensions = &total_ascensions;
    double total_val = 0.0;
    t.total_val = &total_val;
    t.I = &I;
    double best_val = 0.0;
    t.best_val = &best_val;
    std::vector<int> best_x(I.num_vars);
    t.best_x = &best_x;
    std::mutex mtx;
    t.mtx = &mtx;
    for(numrand = 0; numrand <= I.num_vars; numrand++) { //exhaustive hyperparameter sweep (scan over all values of numrand)
        avg = 0.0;
        avg_ascensions = 0.0;
        std::vector<std::thread> threadset(num_threads);
        for(size_t thread_index = 0; thread_index < num_threads; thread_index++) {
            t.seed = seed + num_threads*numrand + thread_index;
            int start, limit;
            dispatcher(TRIALS, num_threads, thread_index, start, limit);
            t.num_trials = limit - start;
            t.num_rand = numrand;
            threadset[thread_index] = std::thread(advrand_trial, t);
        }
        for(size_t thread_index = 0; thread_index < num_threads; thread_index++) threadset[thread_index].join();
        avg = total_val/(double)total_trials;
        avg_ascensions = total_ascensions/(double)total_trials;
        std::cout << "numrand: " << numrand << "/" << I.num_vars << " avg: " << avg << " avg ascensions: " << avg_ascensions << std::endl;
    }
    std::cout << "bestval: " << best_val << std::endl;
    std::cout << (int)std::round(best_val + 0.5*(double)I.num_cons) << " clauses satisfied out of " << I.num_cons << std::endl;
    std::string infile = argv[1];
    std::string outfile = infile + ".sol";
    std::ofstream of(outfile);
    for(int xi : best_x) {
        if(xi == -1) of << "1";
        if(xi == 1) of << "0";
        if(xi != -1 && xi != 1) notify("Error: invalid value in bestx");
    }
    of.close();
    std::cout << "solution saved to " << outfile << std::endl;
    return 0;
}