#ifndef XOROPT_HPP
#define XOROPT_HPP

#include <vector>
#include "bitmatrix.hpp"

class xorsat_instance {
    public:
        //max-XORSAT: satisfy as many as possible of the linear equations: x BT = v
        bitmatrix BT; //B transpose (n by m, in the notation of our paper)
        bitvector v;
        bool load(std::string filename);
        bool save(std::string filename);
        std::string to_string();
        size_t num_vars() const;
        size_t num_cons() const;
};

class walker {
    public:
        walker(const xorsat_instance &instance);
        int value();
        int propose(size_t index);
        void accept();
        void reject();
        void randomize(std::mt19937_64 &eng);
        void set_x(const bitvector &xvals);
        bitvector get_x() const;
        size_t num_vars() const;
        size_t num_cons() const;
    private:
        const xorsat_instance *I;
        bitvector x;
        bitvector y;
        bitvector s;
        bitvector s_proposed;
        size_t flip_proposed;
        int diff_proposed;
        int val;
        bool evaluated;
        bool proposal_queued;
};

//Warning: does not automatically randomize the initial walker position. If you want that do w.randomize() first.
void anneal(walker &w, int iter, double minbeta, double maxbeta, std::mt19937_64 &eng, bool verbose = true);

//This is a greedy algorithm.
void descend(walker &w, bool verbose = true);

int truncation_heuristic(const xorsat_instance &I, bitvector &sol, bool verbose = true);

void sort_instance(xorsat_instance &I, bool ascending, bool verbose = true);

int sort_truncate_descend(const xorsat_instance &I, bitvector &sol, bool reverse, bool verbose);

bool transfer_heaviest(xorsat_instance &from, xorsat_instance &to, size_t threshold);

void merge(const xorsat_instance &from, xorsat_instance &to);

int clauses_violated(const xorsat_instance &I, const bitvector &x);

//Sergei Isakov's algorithm
int partial_anneal(const xorsat_instance &I, size_t weight_cutoff, bitvector &sol, uint64_t seed);

#endif