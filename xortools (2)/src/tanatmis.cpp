#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include "bitmatrix.hpp"
#include "utils.hpp"
#include "xorsat.hpp"
#include "ldpc.hpp"
#include "gurobi_c++.h"

/*
   Here we implement the main algorithm from:
   [1] Akin Tanatmis, Stefan Ruzika, Horst W. Hamacher, Mayur Punekar, Frank Kienle, and Norbert Wehn
       A Separation Algorithm for Improved LP-Decoding of Linear Block Codes
       IEEE Transactions on Information Theory 56, no. 7 (2010): 3277-3289.
*/

//I'm not sure what the threshold should be. I have chosen 1E-4, but we may wish to tune this later.
#define EPSILON 1.0E-4

GRBVar fetchvar(GRBModel *model, std::string name, int i = -1, int j = -1, int k = -1) {
    std::string label = name;
    if(i >= 0) label += "_" + std::to_string(i);
    if(j >= 0) label += "_" + std::to_string(j);
    if(k >= 0) label += "_" + std::to_string(k);
    return model->getVarByName(label);
}

//This is the property that we want \hat{H} to have
bool is_hat(bitmatrix H, std::vector<size_t> nonintegers) {
    size_t bits = H.num_cols();
    for(auto &i : nonintegers) {
        if(i >= bits) {
            notify("Error: index out of range in hhat");
            return false;
        }
        if(H.col_vec(i).count() != 1) return false;
    }
    return true;
}

//This property yields at least one cut
bool is_usable(const bitmatrix &Hhat, const std::vector<size_t> &nonintegerx) {
    size_t checks = Hhat.num_rows();
    size_t bits = Hhat.num_cols();
    bitvector mask, andvec;
    mask.from_sparse(nonintegerx, bits);
    for(size_t i = 0; i < checks; i++) {
        andvec = mask & Hhat.row_vec(i);
        if(andvec.count() == 1) return true;
    }
    return false;
}

//This is a nice way to do the "Construct \hat{H} Algorithm" from [1]
bitmatrix hhat(bitmatrix H, std::vector<size_t>nonintegers) {
    size_t bits = H.num_cols();
    bitmatrix Hhat = H;
    std::vector<bool>used(bits, false);
    std::vector<size_t> permutation;
    bitmatrix P(bits, bits);
    for(auto & i : nonintegers) {
        if(i >= bits) {
            notify("Error: index out of range in hhat");
            return Hhat;
        }
        permutation.push_back(i);
        used[i] = true;
    }
    for(size_t i = 0; i < bits; i++) if(!used[i]) permutation.push_back(i);
    for(size_t i = 0; i < bits; i++) P.set(i,permutation[i],true);
    bitmatrix PI = P.transpose(); //P transpose is P inverse
    Hhat = Hhat * PI;
    size_t rrank = rref(Hhat, nonintegers.size());
    //if(rrank < nonintegers.size()) std::cout << "preperm(" << nonintegers.size() << "):\n" << Hhat << std::endl;
    Hhat = Hhat * P;
    /*if(!is_hat(Hhat, nonintegers)) {
        std::cout << "Hhat failure diagnostics:" << std::endl;
        std::cout << "nonintegers" << vec_string(nonintegers, " ") << std::endl;
        std::cout << "permutation" << vec_string(permutation, " ") << std::endl;
        std::cout << "Hhat:\n" << Hhat << std::endl;
    }*/
    return Hhat;
}

//I thought this might help but it didn't
bitmatrix fancy_hhat(bitmatrix H, std::vector<size_t>nonintegers, std::mt19937_64 &eng) {
    const size_t TRIALS = 20;
    bitmatrix Hhat = hhat(H, nonintegers);
    if(is_usable(Hhat,nonintegers)) return Hhat;
    for(size_t trial = 0; trial < TRIALS; trial++) { 
        std::vector<size_t>perm(H.num_rows());
        for(size_t i = 0; i < H.num_rows(); i++) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), eng);
        bitmatrix PR(H.num_rows(), H.num_rows());
        for(size_t i = 0; i < H.num_rows(); i++) PR.set(i,perm[i],1);
        bitmatrix PRT = PR.transpose();
        Hhat = PRT * hhat(PR * H, nonintegers); //multiplying by PRT makes things tidy but is not necessary
        if(is_usable(Hhat, nonintegers)) {
            std::cout << "Fancy stuff helped: success on trial " << trial << std::endl;
            return Hhat;
        }
    }
    return Hhat;
}

void make_RIPD(GRBModel *model, const bitmatrix &H, const bitvector &c, double p_flip) {
    GRBLinExpr sum;
    size_t bits = H.num_cols();
    size_t checks = H.num_rows();
    if(p_flip < 1.0E-6) {
        p_flip = 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0E-6.");
    }
    if(p_flip > 1.0 - 1.0E-6) {
        p_flip = 1.0 - 1.0E-6;
        notify("Warning: regularizing p_flip to 1.0-1.0E-6.");
    }
    double gamma = log(p_flip/(1.0-p_flip));
    //Nbit[i] is the set of indices for the parity checks neighboring bit i
    std::vector<std::vector<size_t>>Nbit(bits);
    //Ncheck[j] is the set of indices for the bits neighboring parity check j
    std::vector<std::vector<size_t>>Ncheck(checks);
    //T[j] is the set of even numbers from 0 to 2*floor[neighb(j)/2], where neighb(j) is the number of bits neighboring check j
    std::vector<std::vector<size_t>>T(checks);
    for(size_t i = 0; i < bits; i++) {
        bitvector colvec = H.col_vec(i);
        Nbit[i] = colvec.to_sparse();
    }
    for(size_t j = 0; j < checks; j++) {
        bitvector rowvec = H.row_vec(j);
        Ncheck[j] = rowvec.to_sparse();
    }
    //add the variables
    std::string label;
    for(size_t i = 0; i < bits; i++) {
        label = "x_" + std::to_string(i);
        if(c.get(i)) model->addVar(0.0,1.0,gamma,GRB_CONTINUOUS,label);
        else model->addVar(0.0,1.0,-gamma,GRB_CONTINUOUS,label);
    }
    for(size_t j = 0; j < checks; j++) {
        label = "z_" + std::to_string(j);
        model->addVar(0,GRB_INFINITY,0.0,GRB_CONTINUOUS,label);
    }
    model->update(); //so that getVarByName will work in fetchvar
    //add the constraints
    for(size_t j = 0; j < checks; j++) {
        sum = 0;
        for(size_t i : Ncheck[j]) sum += fetchvar(model,"x",i);
        sum -= 2 * fetchvar(model,"z", j);
        model->addConstr(sum == 0);
    }
    model->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    model->update();
    //std::cout << "Objective: " << model->getObjective() << std::endl;
}

bool bitround(double x) {
    if(x >= 0.5) return true;
    return false;
}

bool approx_one(double x) {
    if(fabs(x - 1.0) < EPSILON) return true;
    return false;
}

bool approx_zero(double x) {
    if(fabs(x) < EPSILON) return true;
    return false;
}

bool noninteger(double x) {
    double nearest = std::round(x);
    if(fabs(x - nearest) > EPSILON) return true;
    return false;
}

//it is required that x has size equal to numbe of bits and z has size equal to number of checks
void fetchall(GRBModel *model, std::vector<double> &x, std::vector<double> &z) {
    std::string label;
    for(size_t i = 0; i < x.size(); i++) x[i] = fetchvar(model, "x", i).get(GRB_DoubleAttr_X);
    for(size_t j = 0; j < z.size(); j++) z[j] = fetchvar(model, "z", j).get(GRB_DoubleAttr_X);
}

void cut_generation1(GRBModel *model, const bitmatrix &H, const std::vector<double> &x, const std::vector<double> &z) {
    double k,roundk;
    int intk;
    for(size_t i = 0; i < z.size(); i++) {
        k = 2*z[i];
        roundk = std::round(k);
        if(fabs(k - roundk) > EPSILON) notify("Warning: z is not close to a half-integer.");
        intk = (int)roundk;
        if(intk%2 == 1) {
            std::vector<size_t> N = H.row_vec(i).to_sparse();
            GRBLinExpr sum = 0;
            int Ssize = 0;
            for(size_t j : N) {
                if(approx_one(x[j])) {
                    sum += fetchvar(model, "x", j);
                    Ssize++;
                }
                else sum -= fetchvar(model, "x", j);
            }
            double RHS = (double)(Ssize - 1);
            model->addConstr(sum <= RHS);
        }
    }
}

bool cut_generation2(GRBModel *model, const bitmatrix &Hhat, const std::vector<double> &x, const std::vector<double> &z, const std::vector<size_t> &nonintegerx) {
    size_t checks = Hhat.num_rows();
    size_t bits = Hhat.num_cols();
    bitvector mask, andvec;
    std::vector<size_t>Ni;
    mask.from_sparse(nonintegerx, bits);
    int cuts_added = 0;
    for(size_t i = 0; i < checks; i++) {
        andvec = mask & Hhat.row_vec(i);
        if(andvec.count() == 1) {
            size_t j = andvec.to_sparse()[0];
            cuts_added++;
            int ki = 0;
            Ni = Hhat.row_vec(i).to_sparse();
            GRBLinExpr sum = 0;
            for(size_t h : Ni) {
                if(approx_one(x[h])) sum += fetchvar(model, "x", h);
                if(approx_zero(x[h])) sum -= fetchvar(model, "x", h);
            }
            for(size_t h : Ni) if(approx_one(x[h])) ki++;
            if(ki%2 == 1) model->addConstr(sum - fetchvar(model,"x",j) <= (double)(ki - 1));
            else model->addConstr(sum + fetchvar(model,"x",j) <= (double)ki);
            //In the "cut generation algorithm 2" box they say to terminate upon reaching here for the first time.
            //But in the main text they seem to suggest adding all of the satisfactory cuts you can find.
        }
    }
    if(cuts_added == 0) { //is_hat should succeed whenever rank <= nonintegerx.size() but that's not happening right now!
        std::cout << "nonintegerx.size() = " << nonintegerx.size() << std::endl;
        std::cout << "is_hat = " << is_hat(Hhat, nonintegerx) << std::endl;
        return false; //"error"
    }
    return true; //good
}

//bitvector decode(GRBEnv *env, const bitmatrix &H, const bitvector &c, double p_flip, std::mt19937_64 &eng) {
bitvector decode(GRBEnv *env, const bitmatrix &H, const bitvector &c, double p_flip) {
    const int MAXITER = 1E6;
    size_t checks = H.num_rows();
    size_t bits = H.num_cols();
    bitvector d(bits); //the decoded codeword
    GRBModel *model;
    model = new GRBModel(env);
    model->set(GRB_IntParam_LogToConsole, 0);
    make_RIPD(model, H, c, p_flip);
    model->optimize();
    std::vector<double>x(bits);
    std::vector<double>z(checks);
    std::vector<size_t>nonintegerx;
    std::vector<size_t>nonintegerz;
    bitmatrix Hhat;
    fetchall(model, x, z);
    for(size_t i = 0; i < bits; i++)   if(noninteger(x[i])) nonintegerx.push_back(i);
    for(size_t j = 0; j < checks; j++) if(noninteger(z[j])) nonintegerz.push_back(j);
    size_t iter = 0;
    while(nonintegerx.size() + nonintegerz.size() > 0 && iter < MAXITER) {
        if(nonintegerx.size() > 0) {
            //Hhat = fancy_hhat(H, nonintegerx, eng);
            Hhat = hhat(H, nonintegerx);
            bool success = cut_generation2(model, Hhat, x, z, nonintegerx);
            if(!success) {
                std::cout << "Cut generation algorithm 2 failed." << std::endl;
                break;
            }
        }
        else cut_generation1(model, H, x, z);
        model->optimize();
        fetchall(model, x, z);
        nonintegerx.clear();
        nonintegerz.clear();
        for(size_t i = 0; i < bits; i++)   if(noninteger(x[i])) nonintegerx.push_back(i);
        for(size_t j = 0; j < checks; j++) if(noninteger(z[j])) nonintegerz.push_back(j);
        iter++;
    }
    if(nonintegerx.size() + nonintegerz.size() > 0) std::cout << "Gave up after " << iter << " iterations." << std::endl;
    else std::cout << "Success after " << iter << " iterations" << std::endl;
    //std::cout << "x: " << vec_string(x, " ") << std::endl;
    //std::cout << "z: " << vec_string(z, " ") << std::endl;
    for(size_t i = 0; i < bits; i++) d.set(i,bitround(x[i]));
    delete model;
    return d;
}

void save_results(char *input_name, std::vector<double> success_rate) {
    std::ostringstream buffer;
    buffer << input_name << ".ipout";
    std::string outname = buffer.str();
    std::ofstream outfile(outname);
    for(const auto &r : success_rate) outfile << r << "\t";
    outfile.close();
}

int main(int argc, char *argv[]) {
    if(argc < 2 || argc > 5) {
        std::cout << "Usage: tanatmis instance.tsv <seed> <errors> <trials>" << std::endl;
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
    if(H.num_cols() < 1E5) generator_matrix(H, G); //too costly to compute if H is huge
    GRBEnv env;
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
            //bitvector decoded = decode(&env, H, corrupted_codeword, p_flip, eng);
            bitvector decoded = decode(&env, H, corrupted_codeword, p_flip);
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
                //bitvector decoded = decode(&env, H, corrupted_codeword, p_flip, eng);
                bitvector decoded = decode(&env, H, corrupted_codeword, p_flip);
                std::cout << "c: " << random_codeword << std::endl;
                std::cout << "d: " << decoded << std::endl;
                if(decoded == random_codeword) success++;
            }
            success_rate = (double)success/(double)reps;
            std::cout << success_rate << std::endl;
            if(success_rate >= 0.6) emax = errors;
            success_rates.push_back(success_rate);
            errors++;
        }
        std::cout << std::endl << "emax = " << emax << std::endl;
        save_results(argv[1], success_rates);
    }
    return 0;
}


//This is a test, which has passed
/*int main() {
    bitmatrix H(2,3); //2 cons, 3 vars
    //1->3 repetition code
    H.set(0,0,1);
    H.set(0,1,1);
    H.set(1,1,1);
    H.set(1,2,1);
    bitvector c(3); //all zeros
    c.set(1,1);
    std::cout << H << std::endl;
    GRBEnv env;
    GRBModel *model;
    model = new GRBModel(&env);
    model->set(GRB_IntParam_LogToConsole, 0);
    bitvector d = decode(model, H, c, 0.333333);
    model->write("debug.lp");
    std::cout << "d: " << d << std::endl;
    delete model;
    return 0;
}*/

//This is a test of make_RIPD, which passed
/*int main() {
    bitmatrix H(2,3); //2 cons, 3 vars
    //1->3 repetition code
    H.set(0,0,1);
    H.set(0,1,1);
    H.set(1,1,1);
    H.set(1,2,1);
    bitvector c(3); //all zeros
    c.set(1,1);
    std::cout << H << std::endl;
    GRBEnv env;
    GRBModel *model;
    model = new GRBModel(&env);
    make_RIPD(model, H, c, 0.333333);
    model->write("debug.lp");
    delete model;
    return 0;
}*/

//This is a test of hhat, which passed
/*int main() {
    bitmatrix H(10,20);
    std::random_device rd;
    uint64_t seed = rd();
    std::mt19937_64 eng(seed);
    std::cout << "seed = " << seed << std::endl;
    H.random(eng);
    std::vector <size_t> nonintegers = random_subset<size_t>(20, 5, eng);
    std::cout << "H:\n" << H << std::endl;
    std::cout << "nonintegers: " << vec_string(nonintegers, " ") << std::endl;
    bitmatrix Hhat = hhat(H, nonintegers);
    std::cout << "Hhat:\n" << Hhat << std::endl;
    std::cout << "is_hhat: " << is_hat(Hhat, nonintegers) << std::endl;
    return 0;
}*/