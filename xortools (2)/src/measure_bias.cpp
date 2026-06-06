#include <iostream>
#include <thread>
#include <vector>
#include <unordered_set>
#include "xoropt.hpp"
#include "utils.hpp"

#define TRIALS 1000
#define ITER 10000
#define MAXBETA 4.0

typedef struct {
    uint64_t seed;
    int num_trials;
    xorsat_instance *I;
    std::vector<int> *var_histo;
    std::vector<int> *con_histo;
    std::unordered_set<std::string> *solset;
    std::vector<int> *vals;
    std::mutex *mtx;
}testdata;

void runtests(testdata t) {
    std::mt19937_64 eng(t.seed);
    walker w(*(t.I));
    std::string xstring;
    for(int trial = 0; trial < t.num_trials; trial++) {
        w.randomize(eng);
        anneal(w, ITER, 0.0, MAXBETA, eng, false);
        bitvector s = t.I->v + (w.get_x() * t.I->BT);
        t.mtx->lock();//------------------------------------------------------------------------
        for(size_t i = 0; i < t.I->num_vars(); i++) if(w.get_x().get(i)) t.var_histo->at(i) += 1;
        for(size_t j = 0; j < t.I->num_cons(); j++) if(s.get(j)) t.con_histo->at(j) += 1;
        xstring = w.get_x().to_str();
        if(t.solset->find(xstring) != t.solset->end()) std::cout << "Duplicate found!" << std::endl;
        t.vals->push_back(w.value());
        t.solset->insert(xstring);
        t.mtx->unlock();//----------------------------------------------------------------------
    }
}

typedef struct {
    int min;
    int max;
    double avg;
}valstats;

valstats tabulate_stats(const std::vector<int> &vals) {
    valstats s;
    double total;
    if(vals.size() == 0) return s;
    s.min = vals[0];
    s.max = vals[0];
    total = vals[0];
    for(size_t i = 1; i < vals.size(); i++) {
        if(vals[i] < s.min) s.min = vals[i];
        if(vals[i] > s.max) s.max = vals[i];
        total += vals[i];
    }
    s.avg = total/(double)vals.size();
    return s;
}

void display_results(const testdata &t) {
    std::cout << "variable histogram: " << std::endl;
    std::cout << "index" << "\t" << "degree" << "\t" << "1_frac" << std::endl;
    for(size_t i = 0; i < t.var_histo->size(); i++) std::cout << i << "\t" << t.I->BT.row_vec(i).count() << "\t" << (double)t.var_histo->at(i)/(double)TRIALS << "\n";
    std::cout << "constraint histogram: " << std::endl;
    std::cout << "index" << "\t" << "degree" << "\t" << "sat_frac" << std::endl;
    for(size_t j = 0; j < t.con_histo->size(); j++) std::cout << j << "\t" << t.I->BT.col_vec(j).count() << "\t" << 1.0 - (double)t.con_histo->at(j)/(double)TRIALS << "\n";
    valstats s = tabulate_stats(*(t.vals));
    std::cout << "violated clauses in solutons found: min = " << s.min << " max = " << s.max << " avg = " << s.avg << std::endl;
    std::cout << "avg sat_frac in solutions found: " << 1.0 - s.avg/(double)t.I->num_cons() << std::endl; 
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: measure_bias instance.tsv" << std::endl;
        return 0;
    }
    xorsat_instance I;
    if(!I.load(argv[1])) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    std::cout << "seed: " << seed << std::endl;
    std::vector<int>var_histo(I.num_vars());
    std::vector<int>con_histo(I.num_cons());
    std::vector<int>vals;
    walker w(I);
    //verbose test run--------------------------------------------------------------------------------
    std::cout << "Test run of SA with iter = " << ITER << " and max_beta = " << MAXBETA << std::endl;
    std::mt19937_64 eng(seed);
    w.randomize(eng);
    anneal(w, ITER, 0.0, MAXBETA, eng, true);
    //------------------------------------------------------------------------------------------------
    uint64_t num_threads = std::thread::hardware_concurrency();
    std::cout << "Using " << num_threads << " threads." << std::endl;
    std::vector<std::thread> threadset(num_threads);
    int start, limit;
    std::mutex mut;
    std::unordered_set<std::string> solset;
    testdata t;
    t.I = &I;
    t.var_histo = &var_histo;
    t.con_histo = &con_histo;
    t.vals = &vals;
    t.mtx = &mut;
    t.solset = &solset;
    for(uint64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        dispatcher(TRIALS, num_threads, thread_id, start, limit);
        //std::cout << "thread_id: " << thread_id << " start: " << start << " limit: " << limit << std::endl;
        t.num_trials = limit - start;
        t.seed = seed+thread_id;
        threadset[thread_id] = std::thread(runtests, t);
    }
    for(uint64_t thread_id = 0; thread_id < num_threads; thread_id++) threadset[thread_id].join();
    display_results(t);
    return 0;
}