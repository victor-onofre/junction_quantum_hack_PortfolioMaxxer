#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "bitmatrix.hpp"
#include "utils.hpp"
#include "xorsat.hpp"
#include "ldpc.hpp"
#include "gurobi_c++.h"

bool bitround(double x) {
    if(x >= 0.5) return true;
    return false;
}

bitvector feldman_lp(const bitmatrix &H, const bitvector &c, double p_flip, GRBEnv *env) {
    bitvector d(c.size());
    GRBModel model = GRBModel(*env);
    //model.set(GRB_StringAttr_ModelName, modelname);
    size_t cons = H.num_rows();
    size_t vars = H.num_cols();
    std::vector<GRBVar> varvec;
    double gamma = log(p_flip/(1.0-p_flip));
    std::string varname;
    for(size_t i = 0; i < vars; i++) {
        varname = 'x' + std::to_string(i);
        if(c.get(i)) varvec.push_back(model.addVar(0.0,1.0,gamma,GRB_CONTINUOUS,varname));
        else varvec.push_back(model.addVar(0.0,1.0,-gamma,GRB_CONTINUOUS,varname));
    }
    for(size_t j = 0; j < cons; j++) {
        bitvector convec = H.row_vec(j);
        std::vector<size_t> varlist = convec.to_sparse();
        std::vector<size_t> empty;
        std::vector<std::vector<size_t>> var_omegalist(varlist.size(), empty);
        int D = varlist.size();
        GRBLinExpr sum;
        if(D > 20) {
            std::cout << "Error: D is too big" << std::endl;
            return d;
        }
        int omegamax = 1<<D;
        bitvector omegastring;
        std::vector<size_t>omegalist;
        for(int i = 0; i < omegamax; i++) {
            omegastring.from_num(i,D);
            if(!(omegastring.count() % 2)) {
                std::string label = "omega_" + std::to_string(j) + "_" + omegastring.to_str();
                varvec.push_back(model.addVar(0.0,1.0,0.0,GRB_CONTINUOUS,label));
                size_t omega_index = varvec.size() - 1;
                std::vector<size_t> omega_sparse = omegastring.to_sparse();
                for(const auto &index : omega_sparse) var_omegalist[index].push_back(omega_index);
                omegalist.push_back(omega_index);
            }
        }
        sum = varvec[omegalist[0]];
        for(size_t i = 1; i < omegalist.size(); i++) sum += varvec[omegalist[i]];
        model.addConstr(sum == 1.0); //sum of omegas from this constraint equals 1
        //add the f sum constaints
        for(size_t i = 0; i < varlist.size(); i++) {
            size_t var_index = varlist[i];
            std::vector<size_t> indexlist = var_omegalist[i];
            sum = varvec[indexlist[0]];
            for(size_t i = 1; i < indexlist.size(); i++) sum += varvec[indexlist[i]];
            sum -= varvec[var_index];
            model.addConstr(sum == 0.0);
        }
    }
    //model.update();
    //model.write("debug.lp");
    model.optimize();
    //for(size_t j = 0; j < varvec.size(); j++) std::cout << varvec[j].get(GRB_StringAttr_VarName) << " = " << varvec[j].get(GRB_DoubleAttr_X) << std::endl;
    for(size_t i = 0; i < d.size(); i++) d.set(i,bitround(varvec[i].get(GRB_DoubleAttr_X)));
    return d;
}

int main(int argc, char *argv[]) {
    if(argc != 3 && argc != 4) {
        std::cout << "Usage: feldman_lp instance.tsv num_errors <seed>" << std::endl;
        return 0;
    }
    instance I;
    if(!load_instance(argv[1], I)) return 0;
    size_t errors = std::atoi(argv[2]);
    if(errors > I.num_cons) {
        std::cout << "Error: number of errors cannot be larger than number of bits" << std::endl;
        return 0;
    }
    uint64_t seed;
    if(argc == 4) seed = std::stoull(argv[3]);
    else {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937_64 eng(seed);
    std::string comments;
    comments += "seed = " + std::to_string(seed) + '\n';
    bitmatrix H, G;
    H = parity_check_matrix(I);
    generator_matrix(H, G);
    double p_flip = (double)errors/(double)I.num_cons;
    //make sure the log-likelihoods are not infinite
    if(errors == 0) p_flip = 1.0E-6;
    if(errors == I.num_cons) p_flip = 1.0 - 1.0E-6;
    //generate a random codeword, random error, and lp
    bitvector random_codeword = rand_word(G, eng);
    comments += "p_flip: " + std::to_string(p_flip) + '\n';
    bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
    comments += "Errors:             " + (random_codeword + corrupted_codeword).to_str() + '\n';
    comments += "Corrupted codeword: " + corrupted_codeword.to_str() + '\n';
    comments += "Random codeword:    " + random_codeword.to_str() + '\n';
    GRBEnv env;
    std::string filename = argv[1];
    size_t lastindex = filename.find_last_of("."); 
    bitvector d = feldman_lp(H, corrupted_codeword, p_flip, &env);
    std::cout << comments;
    std::cout << "Decoded to:         " << d << std::endl;
    if(d == random_codeword) std::cout << "Decoding succeeded." << std::endl;
    else std::cout << "Decoding failed." << std::endl;
    return 0;
}
