#include <iostream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
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
    std::cout << "violated clauses in solutions found: min = " << s.min << " max = " << s.max << " avg = " << s.avg << std::endl;
    std::cout << "avg sat_frac in solutions found: " << 1.0 - s.avg/(double)t.I->num_cons() << std::endl; 
}

std::unordered_map<int, std::vector<int>> separate_by_degree(const xorsat_instance &I) {
    std::unordered_map<int, std::vector<int>> map;
    for(size_t j = 0; j < I.num_cons(); j++) {
        int degree = I.BT.col_vec(j).count();
        auto existing = map.find(degree);
        if(existing != map.end()) existing->second.push_back(j);
        else {
            std::vector<int> v;
            v.push_back(j);
            map[degree] = v;
        }
    }
    return map;
}

/*void print_degmap(const std::unordered_map<int, std::vector<int>> &map) {
    std::cout << "degree map: " << std::endl;
    for(auto entry : map) std::cout << entry.first <<  ": " << vec_string(entry.second, ",") << std::endl;\
}*/

std::unordered_map<int, double> avg_sat_by_degree(const std::unordered_map<int, std::vector<int>> &degree_map, const testdata &t) {
    int num, deg;
    double total;
    std::unordered_map<int, double> returnval;
    for(auto entry : degree_map) {
        deg = entry.first;
        total = 0.0;
        num = 0;
        for(auto index : degree_map.at(deg)) {
            total += 1.0 - t.con_histo->at(index)/(double)TRIALS;
            num++;
        }
        double avg = total/(double)num;
        returnval[deg] = avg;
    }
    return returnval;
}

void printmap(const std::unordered_map<int, double> &avmap) {
    for(auto entry : avmap) std::cout << entry.first <<  ": " << entry.second << std::endl;
}

std::unordered_map<int, double> percentile_by_degree(const std::unordered_map<int, std::vector<int>> &degree_map, const testdata &t, double percentile) {
    int deg;
    std::unordered_map<int, double> returnval;
    for(auto entry : degree_map) {
        deg = entry.first;
        std::vector<double> sats(entry.second.size());
        int i = 0;
        for(auto index : degree_map.at(deg)) sats[i++] = 1.0 - t.con_histo->at(index)/(double)TRIALS;
        std::sort(sats.begin(), sats.end());
        //std::cout << "sats: " << vec_string(sats, ",") << std::endl;
        size_t exemplar_index = std::round(percentile*sats.size());
        //std::cout << "exemplar_index: " << exemplar_index << " of " << sats.size() << std::endl;
        returnval[deg] = sats[exemplar_index];
    }
    return returnval;
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: adversarize instance.tsv" << std::endl;
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
    std::unordered_map<int, std::vector<int>> map = separate_by_degree(I);
    //print_degmap(map);
    std::unordered_map<int, double> avmap = avg_sat_by_degree(map, t);
    std::cout << "average sat by degree: " << std::endl;
    printmap(avmap);
    std::unordered_map<int, double> top_quartile = percentile_by_degree(map, t, 0.75);
    std::cout << "top quartile by by degree: " << std::endl;
    printmap(top_quartile);
    std::unordered_map<int, double> bottom_quartile = percentile_by_degree(map, t, 0.25);
    std::cout << "bottom quartile by by degree: " << std::endl;
    printmap(bottom_quartile);
    //now adversarize v:
    int flipcount = 0;
    for(size_t j = 0; j < I.num_cons(); j++) {
        int degree = I.BT.col_vec(j).count();
        double topq = top_quartile[degree];
        double satf = 1.0 - (double)con_histo[j]/(double)TRIALS;
        if(satf >= topq) {
            I.v.flip(j);
            flipcount++;
        }
    }
    std::cout << "Flipped " << flipcount << " signs out of " << I.num_cons() << std::endl;
    std::string inname, outname;
    inname = argv[1];
    outname = "adv_" + inname;
    I.save(outname);
    std::cout << "Wrote adversarized instance to " << outname << std::endl;
    return 0;
}