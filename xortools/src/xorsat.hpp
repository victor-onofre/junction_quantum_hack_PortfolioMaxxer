#ifndef XORSAT_HPP
#define XORSAT_HPP

#include <string>
#include "bitmatrix.hpp"

typedef struct {
    bitvector b; //~b[i] is the RHS of constraint i
    //The following are redundant, but it's good to have the rows and columns both available as bitsets, for performance.
    std::vector <bitvector> c; //c[i][j]: is variable j present in the ith constraint? 1 = yes, 0 = no
    std::vector <bitvector> v; //v[j][i]: is variable j present in the ith constraint? 1 = yes, 0 = no
    bitvector s; //s[i] = 1 if clause i is satisfied, 0 otherwise
} unweighted_instance;

typedef struct {
    //w[i] (always positive) is the weight of instance U[i] 
    std::vector <unweighted_instance> U;
    std::vector <double> w;
    size_t num_vars; //for code readability: same as U[i].v.size()
    size_t num_cons; //total across all weights: i.e. num_cons = sum_i U[i].c.size()
} instance;

bool load_instance(std::string filename, instance &I);

bool save_instance(std::string filename, instance &I);

//evaluate on 100 random bit strings and reject if value is not same
bool mc_equiv(instance &I1, instance &I2, std::mt19937_64 &eng);

std::string to_string(const instance &I);

std::string con_string(const instance &I);

std::string var_string(const instance &I);

//Throughout this codebase our sign convention is that the value is the number of unsatisfied clauses minus
//the number of satisfied clauses. That is, this is a cost function to minimize.
double evaluate(instance &I, const bitvector &x);

//Throughout this codebase our sign convention is that the value is the number of unsatisfied clauses minus
//the number of satisfied clauses. That is, this is a cost function to minimize.
int evaluate(unweighted_instance &UI, const bitvector &x);

//Throughout this codebase our sign convention is that the value is the number of unsatisfied clauses minus
//the number of satisfied clauses. That is, this is a cost function to minimize.
double diff(const instance &I, size_t var_index);

void flip(instance &I, size_t var_index);

void magnitudes(const instance &I, std::vector <double> &M, std::vector <int> &s);

typedef struct {
    double p;
    std::vector <int> f;
    double t;
    double tmag;
}event;

typedef struct {
    double mean_mag;
    double median_mag;
    double max_p_tmag;
}pstats;

bool comp_t(event i, event j);

bool comp_tmag(event i, event j);

void plus_one(std::vector <int> &s);

//the general case
void stats(std::vector <double> M, std::vector <int> s, int l, pstats &st);

void print_pstats(const pstats &s);

//all clauses satisfied means x * parity_check_matrix = target
bitvector target(const instance &I);

//order the instances by weight from highest to lowest
void priority_order(instance &I);

bool is_priority_ordered(const instance &I);

int gaussheur(const bitmatrix &H, const bitvector &t, bitvector &sol);

int descend(instance &I, const bitvector &init, bitvector &sol, bool verbose, bool backward = false);

#endif
