#include <iostream>
#include <vector>
#include "bitmatrix.hpp"
#include "utils.hpp"
#include "gurobi_c++.h"

GRBVar fetchvar(GRBModel *model, std::string name, int i = -1, int j = -1, int k = -1) {
    std::string label = name;
    if(i >= 0) label += "_" + std::to_string(i);
    if(j >= 0) label += "_" + std::to_string(j);
    if(k >= 0) label += "_" + std::to_string(k);
    return model->getVarByName(label);
}

/*
  This implements the linear program defined in appendix II of:
  Using Linear Programming to Decode Binary Linear Codes
  Jon Feldman, Martin J. Wainright, and David R. Karger
  IEEE Transactions on Information Theory Vol 51, No 3, pg 954 (2005)
*/
void polytopeR(const bitmatrix &parity_check_matrix, const bitvector &c, double p_flip) {
    GRBEnv env;
    GRBModel *model;
    GRBLinExpr sum;
    model = new GRBModel(&env);
    if(p_flip < 1.0E-6) {
        p_flip = 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0E-6.");
    }
    if(p_flip > 1.0 - 1.0E-6) {
        p_flip = 1.0 - 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0-1.0E-6.");
    }
    double gamma = log(p_flip/(1.0-p_flip));
    size_t bits = parity_check_matrix.num_cols();
    size_t checks = parity_check_matrix.num_rows();
    //Nbit[i] is the set of indices for the parity checks neighboring bit i
    std::vector<std::vector<size_t>>Nbit(bits);
    //Ncheck[j] is the set of indices for the bits neighboring parity check j
    std::vector<std::vector<size_t>>Ncheck(checks);
    //T[j] is the set of even numbers from 0 to 2*floor[neighb(j)/2], where neighb(j) is the number of bits neighboring check j
    std::vector<std::vector<size_t>>T(checks);
    for(size_t i = 0; i < bits; i++) {
        bitvector colvec = parity_check_matrix.col_vec(i);
        Nbit[i] = colvec.to_sparse();
    }
    for(size_t j = 0; j < checks; j++) {
        bitvector rowvec = parity_check_matrix.row_vec(j);
        Ncheck[j] = rowvec.to_sparse();
    }
    for(size_t j = 0; j < checks; j++) {
        size_t neighb = Ncheck[j].size();
        size_t floorhalfneighb = neighb>>1;
        for(size_t x = 0; x <= floorhalfneighb; x++) T[j].push_back(2*x);
    }
    std::string label;
    for(size_t i = 0; i < bits; i++) {
        label = "f_" + std::to_string(i);
        if(c.get(i)) model->addVar(0.0,1.0,gamma,GRB_CONTINUOUS,label);
        else model->addVar(0.0,1.0,-gamma,GRB_CONTINUOUS,label);
    }
    for(size_t j = 0; j < checks; j++) {
        for(size_t k : T[j]) {
            label = "alpha_" + std::to_string(j) + "_" + std::to_string(k);
            model->addVar(0.0,1.0,0.0,GRB_CONTINUOUS,label);
        }
    }
    for(size_t i = 0; i < bits; i++) {
        for(size_t j : Nbit[i]) {
            for(size_t k : T[j]) {
                label = "z_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k);
                model->addVar(0.0,1.0,0.0,GRB_CONTINUOUS,label);
            }
        }
    }
    model->update(); //so that getVarByName will work in fetchvar
    //next add the constraints
    for(size_t i = 0; i < bits; i++) {
        for(size_t j : Nbit[i]) {
            for(size_t k : T[j]) {
                model->addConstr(fetchvar(model, "z",i,j,k) - fetchvar(model, "alpha",j,k) <= 0.0); //eq 19
            }
        }
    }
    for(size_t i = 0; i < bits; i++) {
        for(size_t j : Nbit[i]) {
            sum = fetchvar(model, "f",i);
            for(size_t k : T[j]) sum -= fetchvar(model, "z",i,j,k);
            model->addConstr(sum == 0.0); //eq 14
        }
    }
    for(size_t j = 0; j < checks; j++) {
        sum = 0.0;
        for(size_t k : T[j]) sum += fetchvar(model, "alpha",j,k);
        model->addConstr(sum == 1.0); //eq 15
    }
    for(size_t j = 0; j < checks; j++) {
        for(size_t k : T[j]) {
            sum = k * fetchvar(model, "alpha",j,k);
            for(size_t i : Ncheck[j]) sum -= fetchvar(model, "z",i,j,k);
            model->addConstr(sum == 0.0); //eq 16
        }
    }
    model->update();
    model->write("R.lp");
    delete model;
}

int main() {
    bitmatrix H(2,3); //2 cons, 3 vars
    //1->3 repetition code
    H.set(0,0,1);
    H.set(0,1,1);
    H.set(1,1,1);
    H.set(1,2,1);
    bitvector c(3); //all zeros
    std::cout << H << std::endl;
    polytopeR(H, c, 0.1);
    return 0;
}