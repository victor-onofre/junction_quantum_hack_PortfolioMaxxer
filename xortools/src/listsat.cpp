#include <string>
#include <fstream>
#include <random>
#include "pmatrix.hpp"
#include "listsat.hpp"
#include "utils.hpp"
#include "modpinverse.hpp"

bool load_lsinstance(std::string filename, lsinstance &I) {
    std::vector<std::string>noncomments;
    std::ifstream infile(filename);
    std::string line;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            noncomments.push_back(line);
        }
    }
    std::vector<std::string>firstline_tokens;
    firstline_tokens = tokenize(noncomments[0], ',');
    if(firstline_tokens.size() != 3) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    int ivars, icons, imodu;
    ivars = stoi(firstline_tokens[0]);
    icons = stoi(firstline_tokens[1]);
    imodu = stoi(firstline_tokens[2]);
    if(ivars < 0 || ivars > 1E7) {
        notify("num_vars(" + std::to_string(ivars) + ") is out of range in " + filename);
        return false;
    }
    if(icons < 0 || icons > 1E7) {
        notify("num_cons(" + std::to_string(icons) + ")  is out of range in " + filename);
        return false;
    }
    if(imodu <= 1 || imodu >= PMAX) {
        notify("modulus (" + std::to_string(imodu) + ") is out of range in " + filename);
        return false;
    }
    if(!p_prime[imodu]) {
        notify("modulus (" + std::to_string(imodu) + ") is not prime in " + filename);
        return false;
    }
    noncomments.erase(noncomments.begin()); //pop off the first line
    if(noncomments.size() != (size_t)icons) {
        notify("Error: number of constraints read (" + std::to_string(noncomments.size()) + ") doesn't match claim (" + std::to_string(icons) + ").");
        return false;
    }
    I.list.clear();
    I.x.clear();
    I.y.clear();
    I.M.clear();
    uint16_t p = (uint16_t)imodu;
    size_t num_vars = (size_t)ivars;
    size_t num_cons = (size_t)icons;
    I.M.set_modulus(p);
    I.x.set_modulus(p);
    I.x.zeros(num_vars);
    I.y.set_modulus(p);
    I.y.zeros(num_cons);
    std::vector<uint16_t> empty;
    for(size_t i = 0; i < num_cons; i++) I.list.push_back(empty);
    pvector col(num_vars, p);
    for(size_t i = 0; i < noncomments.size(); i++) {
        col.zeros(num_vars);
        std::vector<std::string>line_tokens = tokenize(noncomments[i],',');
        for(const auto & token : line_tokens) {
            if(contains(token, ':')) {
                std::vector<std::string> pair = tokenize(token,':');
                if(pair.size() != 2) {
                    notify("Error: misformatted pair in " + filename);
                    return false;
                }
                int loc = stoi(pair[0]);
                int val = stoi(pair[1]);
                if(loc < 0 || loc > (int)num_vars) {
                    notify("Error: location out of range in " + filename);
                    return false;
                }
                if(val < 0 || val >= p) {
                    notify("Error: value out of range in " + filename);
                    return false;
                }
                col.set((size_t)loc, (uint16_t)val);
            }
            else {
                int ibval = stoi(token);
                if(ibval < 0 || ibval > p) {
                    notify("Invalid RHS in constraint " + std::to_string(i) + ": " + noncomments[i]);
                    return false;
                }
                I.list[i].push_back((uint16_t)ibval);
            }
        }
        I.M.append_below(col); //much faster to do this and then transpose
    }
    I.M = I.M.transpose();
    return true;
}

std::string to_string(const lsinstance &I) {
    std::string str = std::to_string(I.M.num_rows()) + "," + std::to_string(I.M.num_cols()) + "," + std::to_string(I.M.p()) + "\n";
    for(size_t j = 0; j < I.M.num_cols(); j++) {
        for(const auto &l : I.list[j]) {
            str += std::to_string(l);
            str += ',';
        }
        pvector col = I.M.col_vec(j);
        str += col.to_sparse_string();
        str += "\n";
    }
    return str;
}

bool save_lsinstance(std::string filename, const lsinstance &I) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()) {
        notify("Unable to write to " + filename);
        return false;
    }
    outfile << to_string(I);
    outfile.close();
    return true;
}

//generate a random sparse instance with probability prob of a given entry in M being nonzero
//and all lists of size p/2
void ls_randsparse(lsinstance &I, uint16_t p, double prob, size_t num_cons, size_t num_vars, std::mt19937_64 &eng) {
    pmatrix M(num_vars, num_cons, p);
    std::bernoulli_distribution coin(prob);
    std::uniform_int_distribution dice1(1,p-1);
    std::uniform_int_distribution dice2(0,p-1);
    for(size_t i = 0; i < M.num_rows(); i++) {
        for(size_t j = 0; j < M.num_cols(); j++) {
            if(coin(eng)) M.set(i,j,dice1(eng));
        }
    }
    for(size_t j = 0; j < num_cons; j++) {
        std::vector <uint16_t> rs = random_subset<uint16_t>(p, p/2, eng);
        I.list.push_back(rs);
    }
    pvector x(num_vars, p); //all zeros
    pvector y(num_vars, p); //all zeros
    I.M = M;
    I.x = x;
    I.y = y;
}

//evaluate on 100 random bit strings and reject if value is not same
bool mc_equiv(lsinstance &I1, lsinstance &I2, std::mt19937_64 &eng) {
    if(I1.M.num_rows() != I2.M.num_rows()) return false;
    if(I1.M.p() != I2.M.p()) return false;
    pvector r(I1.M.num_rows(), I1.M.p());
    for(size_t trials = 0; trials < 100; trials++) {
        r.random(eng);
        if(evaluate(I1, r) != evaluate(I2, r)) return false; //definitely not equivalent
    }
    return true; //might not be equivalent but we didn't find a difference
}

std::string con_string(const lsinstance &I) {
    std::string str;
    for(size_t i = 0; i < I.M.num_rows(); i++) {
        std::vector<entry> conarray = I.M.row_vec(i).to_sparse();
        bool first = true;
        for(const auto &e : conarray) {
            if(!first) str += ",";
            str += std::to_string(e.loc) + ":" + std::to_string(e.val);
            first = false;
        }
    }
    return str;
}

std::string var_string(const lsinstance &I) {
    std::string str;
    for(size_t j = 0; j < I.M.num_cols(); j++) {
        std::vector<entry> vararray = I.M.col_vec(j).to_sparse();
        bool first = true;
        for(const auto &e : vararray) {
            if(!first) str += ",";
            str += std::to_string(e.loc) + ":" + std::to_string(e.val);
            first = false;
        }
    }
    return str;
}

//returns the number of violated constraints
int evaluate(lsinstance &I, const pvector &x) {
    I.x = x;
    I.y = I.x * I.M;
    int violated = 0;
    for(size_t j = 0; j < I.y.size(); j++) if(!contains(I.list[j],I.y.get(j))) violated++;
    return violated;
}

//returns the change in the number of satisfied constraints if we set x[var_index] = val
int diff(const lsinstance &I, size_t var_index, uint16_t val) {
    int change = (int)val - (int)I.x.get(var_index);
    if(change < 0) change += I.x.p();
    int old_eval = 0;
    for(size_t j = 0; j < I.y.size(); j++) if(!contains(I.list[j],I.y.get(j))) old_eval++;
    pvector new_y = I.y + ((uint16_t)change * I.M.row_vec(var_index));
    int new_eval = 0;
    for(size_t j = 0; j < I.y.size(); j++) if(!contains(I.list[j],new_y.get(j))) new_eval++;
    return new_eval - old_eval;
}

//assigns var_index to val and recalculates delta
void assign(lsinstance &I, size_t var_index, uint16_t val) {
    int change = (int)val - (int)I.x.get(var_index);
    I.x.set(var_index, val);
    if(change < 0) change += I.x.p();
    I.y = I.y + ((uint16_t)change * I.M.row_vec(var_index));
}
