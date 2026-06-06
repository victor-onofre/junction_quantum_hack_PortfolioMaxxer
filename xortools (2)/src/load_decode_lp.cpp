#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
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

class lp {
    public:
        lp(const bitmatrix &parity_check_matrix);
        ~lp();
        void printstats();
        bitvector lp_decode(const bitvector &corrupted_codeword, double p_flip);
        int iterations(); //returns the number of iterations used in the most recent decode
        //TODO: implement this later
        //bitvector lp_decode(const bitvector &corrupted_codeword, std::vector <double> p_flip);
    private:
        bitmatrix H;
        bitmatrix HT; //H transpose
        GRBEnv env;
        GRBModel *model;
        size_t vars;
        size_t cons;
        std::vector<GRBVar> varvec;
};

void lp::printstats() {
    int num_lp_vars = model->get(GRB_IntAttr_NumVars);
    int num_lp_cons = model->get(GRB_IntAttr_NumConstrs);
    std::cout << "polytope Q created with: " << std::endl;
    std::cout << num_lp_vars << " LP variables" << std::endl;
    std::cout << num_lp_cons << " LP constraints" << std::endl;
}

lp::lp(const bitmatrix &parity_check_matrix) {
    H = parity_check_matrix;
    HT = parity_check_matrix.transpose();
    model = new GRBModel(&env);
    model->set(GRB_IntParam_LogToConsole, 0);
    vars = H.num_cols();
    cons = H.num_rows();
    std::string varname;
    //as a placeholder, initialize the objective function as if we are given the all-zeros bit string
    //and a bit flip probability of 0.1;
    double gamma = log(0.1/0.9);
    for(size_t i = 0; i < vars; i++) {
        varname = 'x' + std::to_string(i);
        varvec.push_back(model->addVar(0.0,1.0,-gamma,GRB_CONTINUOUS,varname));
    }
    //create the rest of the model from H.
    for(size_t j = 0; j < cons; j++) {
        bitvector convec = H.row_vec(j);
        std::vector<size_t> varlist = convec.to_sparse();
        std::vector<size_t> empty;
        std::vector<std::vector<size_t>> var_omegalist(varlist.size(), empty);
        int D = varlist.size();
        GRBLinExpr sum;
        if(D > 20) {
            std::cout << "Error: D is too big" << std::endl;
            return;
        }
        int omegamax = 1<<D;
        bitvector omegastring;
        std::vector<size_t>omegalist;
        for(int i = 0; i < omegamax; i++) {
            omegastring.from_num(i,D);
            if(!(omegastring.count() % 2)) {
                std::string label = "omega_" + std::to_string(j) + "_" + omegastring.to_str();
                varvec.push_back(model->addVar(0.0,1.0,0.0,GRB_CONTINUOUS,label));
                size_t omega_index = varvec.size() - 1;
                std::vector<size_t> omega_sparse = omegastring.to_sparse();
                for(const auto &index : omega_sparse) var_omegalist[index].push_back(omega_index);
                omegalist.push_back(omega_index);
            }
        }
        sum = varvec[omegalist[0]];
        for(size_t i = 1; i < omegalist.size(); i++) sum += varvec[omegalist[i]];
        model->addConstr(sum == 1.0); //sum of omegas from this constraint equals 1
        //add the f sum constaints
        for(size_t i = 0; i < varlist.size(); i++) {
            size_t var_index = varlist[i];
            std::vector<size_t> indexlist = var_omegalist[i];
            sum = varvec[indexlist[0]];
            for(size_t i = 1; i < indexlist.size(); i++) sum += varvec[indexlist[i]];
            sum -= varvec[var_index];
            model->addConstr(sum == 0.0);
        }
    }
    model->update();
}

lp::~lp() {
    delete model;
}

bitvector lp::lp_decode(const bitvector &corrupted_codeword, double p_flip) {
    bitvector d(corrupted_codeword.size());
    GRBLinExpr obj;
    if(p_flip < 1.0E-6) {
        p_flip = 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0E-6.");
    }
    if(p_flip > 1.0 - 1.0E-6) {
        p_flip = 1.0 - 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0-1.0E-6.");
    }
    double gamma = log(p_flip/(1.0-p_flip));
    for(size_t i = 0; i < vars; i++) {
        if(corrupted_codeword.get(i)) obj += gamma * varvec[i];
        else obj -= gamma * varvec[i];
    }
    model->setObjective(obj, GRB_MINIMIZE);
    model->optimize();
    for(size_t i = 0; i < d.size(); i++) d.set(i,bitround(varvec[i].get(GRB_DoubleAttr_X)));
    return d;
}

int lp::iterations() {
    return (int)model->get(GRB_DoubleAttr_IterCount);
}

void save_results(char *input_name, std::vector<double> success_rate) {
    std::ostringstream buffer;
    buffer << input_name << ".lpout";
    std::string outname = buffer.str();
    std::ofstream outfile(outname);
    for(const auto &r : success_rate) outfile << r << "\t";
    outfile.close();
}

int main(int argc, char *argv[]) {
    if(argc < 2 || argc > 5) {
        std::cout << "Usage: load_decode instance.tsv <seed> <errors> <trials>" << std::endl;
        return 0;
    }
    instance I;
    if(!load_instance(argv[1], I)) return 0;
    std::random_device rd;
    uint64_t seed = rd();
    if(argc >= 3) seed = std::stoull(argv[2]);
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    bitmatrix H, G;
    H = parity_check_matrix(I);
    lp dec(H);
    dec.printstats();
    if(H.num_cols() < 1E5) generator_matrix(H, G); //too costly to compute if H is huge
    std::vector<double> success_rates;
    int success;
    double success_rate = 1.0;
    double p_flip;
    int emax = 0;
    size_t errors = 0;
    bitvector random_codeword;
    int reps = 10;
    if(argc == 5) {
        int raw = std::stoi(argv[4]);
        if(raw < 1 || raw > 1E6) {
            std::cout << "Invalid reps" << std::endl;
            return 0;
        }
        reps = raw;
    }
    if(argc >= 4) {
        int raw = std::stoi(argv[3]);
        if(raw < 0 || raw > (int)H.num_cols()) {
            std::cout << "Invalid error count." << std::endl;
            return 0;
        }
        errors = (size_t)raw;
        success = 0;
        p_flip = (double)errors/(double)H.num_cols();
        for(int repetition = 0; repetition < reps; repetition++) {
            if(H.num_cols() < 1E5) random_codeword = rand_word(G, eng);
            else random_codeword.zeros(H.num_cols()); //not so random, but definitely a codeword
            bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
            bitvector decoded = dec.lp_decode(corrupted_codeword, p_flip);
            if(decoded == random_codeword) success++;
        }
        success_rate = (double)success/(double)reps;
        std::cout << argv[1] << "\t" << seed << "\t" << errors << "\t" << success_rate << std::endl;
    }
    else {
        while(errors < H.num_cols() && success_rate >= 0.5) {
            success = 0;
            std::cout << errors << ": ";
            p_flip = (double)errors/(double)H.num_cols();
            int max_success_iter = 0;
            for(int repetition = 0; repetition < reps; repetition++) {
                if(H.num_cols() < 1E5) random_codeword = rand_word(G, eng);
                else random_codeword.zeros(H.num_cols()); //not so random, but definitely a codeword
                bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
                bitvector decoded = dec.lp_decode(corrupted_codeword, p_flip);
                if(decoded == random_codeword) {
                    success++;
                    if(dec.iterations() > max_success_iter) max_success_iter = dec.iterations();
                }
            }
            success_rate = (double)success/(double)reps;
            std::cout << success_rate << "\t(" << max_success_iter << ")" << std::endl;
            if(success_rate >= 0.6) emax = errors;
            success_rates.push_back(success_rate);
            errors++;
        }
        std::cout << std::endl << "emax = " << emax << std::endl;
        save_results(argv[1], success_rates);
    }
    return 0;
}
