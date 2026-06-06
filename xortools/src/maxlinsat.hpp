#ifndef MAXLINSAT_HPP
#define MAXLINSAT_HPP

#include <vector>
#include <cstdint>
#include <random>
#include <fstream>
#include <iostream>
#include "utils.hpp"

//For simulated annealing we don't need the modulus to be prime and we don't need to compute inverses mod p.
//So there is no need to limit the modulus to the range treated in modpinverse.hpp, which currently tops out at 512.

typedef struct {
    std::vector<std::vector<bool>> f; //f[i] is the truth table of constraint i
    std::vector <std::vector<uint64_t>> v; //v[j][j]: coefficient of variable j in ith constraint
    std::vector<uint64_t> x;
    std::vector<uint64_t> y; //y = x B
    size_t num_vars;
    size_t num_cons;
    uint64_t modulus;
} modinstance;

void print_token_array(const std::vector <std::vector <std::string>> &token_array) {
    for(const std::vector <std::string> & line : token_array) {
        for(const std::string &token : line) {
            std::cout << token << ' ';
        }
        std::cout << '\n';
    }
}

bool load_modinstance(std::string filename, modinstance &I) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> firstline_tokens;
    std::vector <std::string> line_tokens;
    std::string line;
    int num_vars, num_cons, modulus;
    //if the instance is not empty then empty it------
    I.v.clear();
    I.x.clear();
    I.f.clear();
    I.num_vars = 0;
    I.num_cons = 0;
    //------------------------------------------------
    std::ifstream infile(filename);
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return false;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            line_tokens = tokenize(line, ',');
            token_array.push_back(line_tokens);
        }
    }
    infile.close();
    if(token_array[0].size() != 3) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    num_vars = stoi(token_array[0][0]);
    num_cons = stoi(token_array[0][1]);
    modulus  = stoi(token_array[0][2]);
    if(num_vars < 0 || num_vars > 1E7) {
        notify("num_vars(" + std::to_string(num_vars) + ") is out of range in " + filename);
        return false;
    }
    if(num_cons < 0 || num_cons > 1E7) {
        notify("num_cons(" + std::to_string(num_cons) + ")  is out of range in " + filename);
        return 0;
    }
    if(modulus < 2 || modulus > 1E7) {
        notify("modulus(" + std::to_string(modulus) + ")  is out of range in " + filename);
        return 0;
    }
    token_array.erase(token_array.begin()); //pop off the first line
    if(token_array.size() != (size_t)num_cons) {
        notify("Error: number of constraints read (" + std::to_string(token_array.size()) + ") doesn't match claim (" + std::to_string(num_cons) + ").");
        return false;
    }
    std::vector<std::vector<uint64_t>> v(num_vars, std::vector<uint64_t>(num_cons,0));
    std::vector<std::vector<bool>> f(num_cons, std::vector<bool>(modulus, false));
    int constraint_index = 0;
    int val, loc;
    for(const std::vector<std::string> &line : token_array) {
        for(const std::string &token : line) {
            std::vector<std::string> pair = tokenize(token, ':');
            switch(pair.size()) {
                case 1: //its a truth table entry
                    val = stoi(pair[0]);
                    if(val < 0 || val >= modulus) notify("Invalid list value: " + std::to_string(val));
                    else f[constraint_index][val] = true;
                    break;
                case 2: //its a coefficient
                    loc = stoi(pair[0]);
                    val = stoi(pair[1]);
                    if(loc < 0 || loc > num_vars) {
                        notify("Invalid location: " + std::to_string(loc));
                        break;
                    }
                    if(val < 0 || val >= modulus) {
                        notify("Invalid coefficient: " + std::to_string(val));
                        break;
                    }
                    v[loc][constraint_index] = val;
                    break;
                default:
                    notify("Invalid token: " + token);
            }
        }
        constraint_index++;
    }
    I.v = v;
    I.f = f;
    std::vector<uint64_t> y(num_cons,0);
    std::vector<uint64_t> x(num_vars, 0);
    I.x = x;
    I.y = y;
    I.modulus = modulus;
    I.num_vars = num_vars;
    I.num_cons = num_cons;
    return true;
}

std::string modinstance_string(const modinstance &I) {
    std::string returnval;
    returnval = std::to_string(I.num_vars) + ',' + std::to_string(I.num_cons) + ',' + std::to_string(I.modulus) + '\n';
    for(size_t j = 0; j < I.num_cons; j++) {
        bool first = true;
        for(uint64_t val = 0; val < I.modulus; val++) {
            if(I.f[j][val]) {
                if(!first) returnval += ',';
                returnval += std::to_string(val);
                first = false;
            }
        }
        for(size_t i = 0; i < I.num_vars; i++) {
            if(I.v[i][j] != 0) returnval += ',' + std::to_string(i) + ':' + std::to_string(I.v[i][j]);
        }
        returnval += '\n';
    }
    return returnval;
}

bool save_modinstance(std::string filename, const modinstance &I) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()) {
        notify("Unable to write to " + filename);
        return false;
    }
    outfile << modinstance_string(I);
    outfile.close();
    return true;
}

//returns the number of violated constraints
//does not update I.x or I.y
int evaluate_at(const modinstance &I, const std::vector<uint64_t> &x) {
    if(x.size() != I.num_vars) {
        notify("Invalid var vector in evaluate");
        return 0;
    }
    std::vector<uint64_t> y(I.num_cons, 0);
    int violated = 0;
    for(size_t i = 0; i < I.num_vars; i++) {
        if(x[i] != 0) {
            const std::vector<uint64_t> & varcoeffs = I.v[i];
            for(size_t j = 0; j < I.num_cons; j++) {
                y[j] += x[i] * varcoeffs[j];
            }
        }
    }
    for(size_t j = 0; j < I.num_cons; j++) y[j] = y[j] % I.modulus;
    for(size_t j = 0; j < I.num_cons; j++) {
        if(!I.f[j][y[j]]) violated++;
    }
    return violated;
}

//returns the number of violated constraints at I.x 
//updates I.y
int evaluate(modinstance &I) {
    int violated = 0;
    std::fill(I.y.begin(), I.y.end(),0);
    for(size_t i = 0; i < I.num_vars; i++) {
        if(I.x[i] != 0) {
            const std::vector<uint64_t> & varcoeffs = I.v[i];
            for(size_t j = 0; j < I.num_cons; j++) {
                I.y[j] += I.x[i] * varcoeffs[j];
            }
        }
    }
    for(size_t j = 0; j < I.num_cons; j++) I.y[j] = I.y[j] % I.modulus;
    for(size_t j = 0; j < I.num_cons; j++) {
        if(!I.f[j][I.y[j]]) violated++;
    }
    return violated;
}

//returns the change in the number of violated constraints if we set x[var_index] = val
int diff(const modinstance &I, size_t var_index, uint64_t val) {
    if(val == I.x[var_index]) return 0;
    uint64_t minusold = I.modulus - I.x[var_index];
    uint64_t diff = (val + minusold) % I.modulus;
    int oldsat, newsat;
    uint64_t newval;
    const std::vector<uint64_t> & varcoeffs = I.v[var_index];
    int delta = 0;
    for(size_t j = 0; j < I.num_cons; j++) {
        oldsat = I.f[j][I.y[j]];
        newval = (I.y[j] + diff * varcoeffs[j]) % I.modulus;
        newsat = I.f[j][newval];
        delta -= (newsat - oldsat); //we are counting violated clauses, not satisfied clauses, hence the sign
    }
    return delta;
}

void assign(modinstance &I, size_t var_index, uint64_t val) {
    if(val == I.x[var_index]) return;
    uint64_t minusold = I.modulus - I.x[var_index];
    uint64_t diff = (val + minusold) % I.modulus;
    I.x[var_index] = val;
    const std::vector<uint64_t> & varcoeffs = I.v[var_index];
    for(size_t j = 0; j < I.num_cons; j++) {
        I.y[j] = (I.y[j] + diff * varcoeffs[j]) % I.modulus;
    }
}

void randomize_pvec(std::vector<uint64_t> &x, int modulus, std::mt19937_64 &eng) {
    std::uniform_int_distribution<uint64_t> distr(0, modulus);
    for(size_t i = 0; i < x.size(); i++) x[i] = distr(eng);
}

std::string pvec_string(const std::vector<uint64_t> &v) {
    std::string returnval;
    bool first = true;
    for(const uint64_t & val : v) {
        if(!first) returnval += ',';
        returnval += std::to_string(val);
        first = false;
    }
    return returnval;
}

std::vector<uint64_t> from_str(std::string s, int modulus) {
    std::vector<uint64_t> x;
    if(s.length() == 0) {
        notify("Error: empty string in pvector::from_str");
        return x;
    }
    std::vector <std::string> tokens = tokenize(s, ',');
    x.resize(tokens.size());
    for(size_t i = 0; i < tokens.size(); i++) {
        int val = stoi(tokens[i]);
        if(val < 0 || val >= modulus) {
            notify("Error: invalid input to from_str");
            return x;
        }
        x[i] = (uint64_t)val;
    }
    return x;
}

std::vector<uint64_t> load_solution(std::string filename, int modulus) {
    std::ifstream infile(filename);
    std::string line;
    std::vector <std::string> noncomments;
    std::vector<uint64_t> x;
    if(!infile.is_open()) {
        notify("Unable to open " + filename);
        return x;
    }
    while(getline(infile,line)) {
        if(line[0] != '#') { //skip comment lines
            noncomments.push_back(line);
        }
    }
    if(noncomments.size() > 2) {
        notify("Error: more than two noncomment lines in " + filename);
        return x;
    }
    infile.close();
    int ip, in;
    std::vector<std::string> tokens = tokenize(noncomments[0],',');
    if(tokens.size() != 2) {
        notify("Error: misformatted first noncomment line in " + filename);
        return x;
    }
    ip = stoi(tokens[0]);
    in = stoi(tokens[1]);
    if(ip != modulus) notify("Unexpected modulus in " + filename);
    x = from_str(noncomments[1],modulus);
    if((int)x.size() != in) notify("Error: number of symbols read does not match claim in " + filename);
    return x;
}

#endif