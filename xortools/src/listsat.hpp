#ifndef LISTSAT_HPP
#define LISTSAT_HPP

#include <string>
#include "pmatrix.hpp"

typedef struct {
    std::vector<std::vector<uint16_t> >list;
    pmatrix M;
    pvector x;
    pvector y; //y = xM
} lsinstance;

bool load_lsinstance(std::string filename, lsinstance &I);

bool save_lsinstance(std::string filename, const lsinstance &I);

//generate a random sparse instance with probability prob of a given entry in M being nonzero
//and all lists of size p/2
void ls_randsparse(lsinstance &I, uint16_t p, double prob, size_t num_cons, size_t num_vars, std::mt19937_64 &eng);

//evaluate on 100 random bit strings and reject if value is not same
bool mc_equiv(lsinstance &I1, lsinstance &I2, std::mt19937_64 &eng);

std::string to_string(const lsinstance &I);

std::string con_string(const lsinstance &I);

std::string var_string(const lsinstance &I);

//returns the number of violated constraints
int evaluate(lsinstance &I, const pvector &x);

//returns the change in the number of satisfied constraints if we set x[var_index] = val
int diff(const lsinstance &I, size_t var_index, uint16_t val);

void assign(lsinstance &I, size_t var_index, uint16_t val);

#endif
