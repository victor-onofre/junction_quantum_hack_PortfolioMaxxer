#ifndef ADVRAND_HPP
#define ADVRAND_HPP

#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include "utils.hpp"

class constraint {
    public:
        double coeff;
        size_t num_vars;
        std::set<size_t>index;
        std::string to_string() const;
        bool from_string(const std::string &s, size_t num_vars);
        double value(const std::vector<int> &x) const;
};

class sparse_xorsat {
    public:
        size_t num_vars;
        size_t num_cons;
        std::vector<constraint> c;
        std::string to_string() const;
        void save(std::string filename) const;
        bool load(std::string filename);
        double value(const std::vector<int> &x);
};

bool constraint::from_string(const std::string &s, size_t numvars) {
    std::vector<std::string> token_array = tokenize(s, '\t');
    if(token_array.size() < 2) return false;
    coeff = std::stod(token_array[0]);
    for(size_t token_index = 1; token_index < token_array.size(); token_index++) {
        int var_index = std::stoi(token_array[token_index]);
        if(var_index < 0 || (size_t)var_index >= numvars) return false;
        index.insert(var_index);
    }
    num_vars = numvars;
    return true;
}

std::string constraint::to_string() const {
    std::string s;
    s = to_digits(coeff, 4);
    for(size_t v : index) s += '\t' + std::to_string(v);
    return s;
}

double constraint::value(const std::vector<int> &x) const {
    int ival = 1;
    if(x.size() != num_vars) {
        notify("Wrong input size to constraint::value");
        return 0.0;
    }
    for(size_t var_index : index) ival *= x[var_index];
    return coeff*(double)ival;
}

bool sparse_xorsat::load(std::string filename) {
    std::vector <std::string> lines;
    std::vector <std::string> line_tokens;
    std::string line;
    int inum_vars, inum_cons;
    std::ifstream infile(filename);
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            lines.push_back(line);
        }
    }
    infile.close();
    line_tokens = tokenize(lines[0], '\t');
    if(line_tokens.size() != 2) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    inum_vars = stoi(line_tokens[0]);
    inum_cons = stoi(line_tokens[1]);
    if(inum_vars < 0 || inum_vars > 1E7) {
        notify("num_vars(" + std::to_string(inum_vars) + ") is out of range in " + filename);
        return false;
    }
    if(inum_cons < 0 || inum_cons > 1E7) {
        notify("num_cons(" + std::to_string(inum_cons) + ")  is out of range in " + filename);
        return 0;
    }
    lines.erase(lines.begin()); //pop off the first line
    if(lines.size() != (size_t)inum_cons) {
        notify("Error: number of constraints read (" + std::to_string(lines.size()) + ") doesn't match claim (" + std::to_string(inum_cons) + ").");
        return false;
    }
    for(const std::string &l : lines) {
        constraint con;
        if(con.from_string(l, inum_vars)) c.push_back(con);
        else {
            notify("Misformatted line: " + l);
            return false;
        }
    }
    num_vars = (size_t)inum_vars;
    num_cons = (size_t)inum_cons;
    return true;
}

std::string sparse_xorsat::to_string() const {
    std::string s = std::to_string(num_vars) + '\t' + std::to_string(num_cons) + '\n';
    for(const constraint & con : c) s += con.to_string() + '\n';
    return s;
}

void sparse_xorsat::save(std::string filename) const {
    std::ofstream outfile(filename);
    outfile << to_string();
    outfile.close();
}

double sparse_xorsat::value(const std::vector<int> &x) {
    double val = 0.0;
    for(const constraint &con : c) val += con.value(x);
    return val;
}

std::vector<int> load_sol(std::string filename) {
    std::ifstream infile(filename);
    std::string line;
    std::vector <std::string> noncomments;
    size_t char_index;
    std::vector<int> x;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return x;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            noncomments.push_back(line);
        }
    }
    if(noncomments.size() > 1) {
        notify("Error: more than one noncomment line in " + filename);
        return x;
    }
    for(size_t i = 0; i < noncomments[0].length(); i++) {
        if(noncomments[0][i] != '0' && noncomments[0][i] != '1') {
            notify("Error: invalid character in bitvector::load_dense");
            return x;
        }
    }
    x = std::vector<int>(noncomments[0].size(),0);
    for(char_index = 0; char_index < noncomments[0].length(); char_index++) {
        if(noncomments[0][char_index] == '1') x[char_index] = -1;
        if(noncomments[0][char_index] == '0') x[char_index] = +1;
    }
    infile.close();
    return x;
}

//fills in partial derivatives for all variables equal to zero
//leaves the rest of the entries zero
void partial_gradient(const sparse_xorsat &I, const std::vector<int> &x, std::vector <double> &grad) {
    fill(grad.begin(), grad.end(), 0.0);
    size_t zeros, zero_index;
    int product;
    for(const constraint &con : I.c) {
        zeros = 0;
        product = 1;
        for(size_t var_index : con.index) {
            if(x[var_index] == 0) {
                zeros++;
                zero_index = var_index;
            }
            else product *= x[var_index];
            if(zeros > 1) break;
        }
        if(zeros == 1) grad[zero_index] += con.coeff * product;
    }
}

void partial_random(std::vector<int> &x, size_t num, std::mt19937_64 &eng) {
    std::bernoulli_distribution coin(0.5);
    if(num > x.size()) {
        notify("Error: can't randomize more than the number of entries.");
        return;
    }
    std::vector<size_t> rs = random_subset<size_t>(x.size(), num, eng);
    int val[2] = {1, -1};
    std::fill(x.begin(), x.end(), 0);
    for(size_t index : rs) x[index] = val[coin(eng)];
}

int ascend(std::vector<int> &x, std::vector<double>grad, std::mt19937_64 &eng) {
    if(x.size() != grad.size()) {
        notify("Error: Sizes don't match in ascend.");
        return 0;
    }
    int ascended = 0;
    std::bernoulli_distribution coin(0.5);
    int val[2] = {1,-1};
    for(size_t i = 0; i < x.size(); i++) {
        if(x[i] == 0) {
            if(grad[i] > 0) {
                x[i] = 1;
                ascended++;
            }
            if(grad[i] < 0) {
                x[i] = -1;
                ascended++;
            }
            if(grad[i] == 0) x[i] = val[coin(eng)];
        }
    }
    return ascended;
}

int descend(std::vector<int> &x, std::vector<double>grad, std::mt19937_64 &eng) {
    if(x.size() != grad.size()) {
        notify("Error: Sizes don't match in descend.");
        return 0;
    }
    int descended = 0;
    std::bernoulli_distribution coin(0.5);
    int val[2] = {1,-1};
    for(size_t i = 0; i < x.size(); i++) {
        if(x[i] == 0) {
            if(grad[i] > 0) {
                x[i] = -1;
                descended++;
            }
            if(grad[i] < 0) {
                x[i] = 1;
                descended++;
            }
            if(grad[i] == 0) x[i] = val[coin(eng)];
        }
    }
    return descended;
}

void perturb(std::vector<int> &x, double pflip, std::mt19937_64 &eng) {
    if(pflip < 0 || pflip > 1.0) {
        notify("Error invalid pflip");
        return;
    }
    std::bernoulli_distribution coin(pflip);
    for(size_t i = 0; i < x.size(); i++) if(coin(eng)) x[i] *= -1;
}

#endif