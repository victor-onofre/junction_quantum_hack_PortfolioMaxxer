#include <iostream>
#include <random>
#include <thread>
#include "xoropt.hpp"
#include "utils.hpp"

#define TRIALS 200

std::mutex mtx;

//Sergei Isakov's algorithm, adapted for a multithreaded context
void accumulate_partial_anneal(const xorsat_instance &I, size_t weight_cutoff, bitvector *sol, uint64_t seed, int reps) {
    std::mt19937_64 eng(seed);
    xorsat_instance Iheavy, Ilight;
    Ilight = I;
    transfer_heaviest(Ilight, Iheavy, weight_cutoff);
    walker w1(Ilight);
    walker w2(I);
    bitvector x;
    for(int trial = 0; trial < reps; trial++) {
        /*mtx.lock();
        std::cout << "seed: " << seed << " trial: " << trial << std::endl;
        mtx.unlock();*/
        w1.randomize(eng);
        anneal(w1, 1000000, 0.5, 4.0, eng, false);
        x = w1.get_x();
        w2.set_x(x);
        int violated_before_descent = w2.value();
        descend(w2, false);
        int violated_after_descent = w2.value();
        mtx.lock();//----------------------------------------------------------------------------
        std::cout << violated_before_descent << " -> " << violated_after_descent << std::endl;
        int oldval = clauses_violated(I, *sol);
        if(w2.value() < oldval) *sol = w2.get_x();
        mtx.unlock();//--------------------------------------------------------------------------
    }
}

//Sergei Isakov's parameters:
int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: partial_anneal_mp instance.tsv" << std::endl;
        return 0;
    }
    xorsat_instance I;
    if(!I.load(argv[1])) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    //std::mt19937_64 eng(seed);
    bitvector x(I.num_vars());
    uint64_t num_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " threads.\n";
    std::thread threadset[num_threads];
    int start, limit;
    for(uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        dispatcher(TRIALS, num_threads, thread_id, start, limit);
        threadset[thread_id] = std::thread(accumulate_partial_anneal, I, 21, &x, seed+thread_id, limit-start);
    }
    for(uint64_t thread_id = 0; thread_id < num_threads; thread_id++) threadset[thread_id].join();
    int best_val = clauses_violated(I, x);
    std::cout << "Best val: " << best_val << std::endl;
    std::cout << I.num_cons() - best_val << " clauses satisfied out of " << I.num_cons() << std::endl;
    std::cout << "Best x:\n" << x << std::endl;
    return 0;
}