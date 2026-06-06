#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include "xorsat.hpp"
#include "utils.hpp"

void separate(const std::vector <std::vector <std::string> > &token_array, std::vector <std::vector <std::vector <std::string> > > &instance_tokens, std::vector <double> &weights) {
    std::vector <double> raw_weights;
    double tolerance = 1E-4;
    for(std::vector <std::string> const &line : token_array) raw_weights.push_back(stod(line[0]));
    for(double const &weight : raw_weights) {
        double mag = fabs(weight);
        if(!approx_present(weights, mag, tolerance)) weights.push_back(mag);
    }
    //allocate outmost vector---------------------------------
    std::vector <std::vector <std::vector <std::string> > > it(weights.size()); 
    instance_tokens = it;
    //--------------------------------------------------------
    for(std::vector <std::string> const &line : token_array) {
        size_t i = approx_find(weights, fabs(stod(line[0])), tolerance);
        instance_tokens[i].push_back(line);
    }
}

bool parse_unsigned(const std::vector <std::vector <std::string> > &token_array, unweighted_instance &UI, size_t num_vars) {
    for(std::vector <std::string> const &line_tokens : token_array) {
        double weight = stod(line_tokens[0]);
        if(weight == 0) {
            notify("Error: zero weight.");
            return false;
        }
        if(weight > 0) UI.b.push_back(0);
        if(weight < 0) UI.b.push_back(1);
    }
    //allocate v and c and intialize to all zeros--------------------------------
    size_t num_cons = UI.b.size(); //just for code readability
    bitvector zero_varstring(num_cons);
    bitvector zero_constring(num_vars);
    std::vector <bitvector> v_zero(num_vars,zero_varstring);
    std::vector <bitvector> c_zero(num_cons,zero_constring);
    UI.v = v_zero;
    UI.c = c_zero;
    //---------------------------------------------------------------------
    int var_index;
    int con_index = 0;
    for(std::vector <std::string> const &line_tokens : token_array) {
        bitvector constraint_bitstring(num_vars); //initialize to zeros
        for(size_t tokennum = 1; tokennum < line_tokens.size(); tokennum++) {
            var_index = stoi(line_tokens[tokennum]);
            if(var_index < 0 || (size_t)var_index > num_vars) {
                notify("Error: index (" + std::to_string(var_index) + ") is out of range [0," + std::to_string(num_vars) + "].");
                return false;
            }
            UI.c[con_index].set(var_index, 1);
            UI.v[var_index].set(con_index, 1);
        }
        con_index++;
    }
    return true;
}

bool load_instance(std::string filename, instance &I) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> firstline_tokens;
    std::vector <std::string> line_tokens;
    std::string line;
    int num_vars, num_cons;
    //if the instance is not empty then empty it------
    I.U.clear();
    I.w.clear();
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
            line_tokens = tokenize(line, '\t');
            token_array.push_back(line_tokens);
        }
    }
    infile.close();
    if(token_array[0].size() != 2) {
        notify("First line of " + filename + " is misformatted.");
        return false;
    }
    num_vars = stoi(token_array[0][0]);
    num_cons = stoi(token_array[0][1]);
    if(num_vars < 0 || num_vars > 1E7) {
        notify("num_vars(" + std::to_string(num_vars) + ") is out of range in " + filename);
        return false;
    }
    if(num_cons < 0 || num_cons > 1E7) {
        notify("num_cons(" + std::to_string(num_cons) + ")  is out of range in " + filename);
        return 0;
    }
    token_array.erase(token_array.begin()); //pop off the first line
    if(token_array.size() != (size_t)num_cons) {
        notify("Error: number of constraints read (" + std::to_string(token_array.size()) + ") doesn't match claim (" + std::to_string(num_cons) + ").");
        return false;
    }
    std::vector < std::vector < std::vector <std::string> > > instance_tokens;
    std::vector < double > weights;
    separate(token_array, instance_tokens, weights);
    bool success;
    for(size_t i = 0; i < weights.size(); i++) {
        unweighted_instance UI;
        success = parse_unsigned(instance_tokens[i], UI, num_vars);
        if(!success) return false;
        I.U.push_back(UI);
        I.w.push_back(weights[i]);
    }
    size_t total_cons = 0;
    for(unweighted_instance &UI : I.U) {
        bitvector s_local(UI.c.size());
        UI.s = s_local;            //allocate s
        total_cons += UI.c.size(); //tally up the total constraint count
    }
    I.num_cons = total_cons;
    I.num_vars = I.U[0].v.size();
    return true;
}

std::string to_string(const instance &I) {
    int sign;
    double weight;
    std::string str = std::to_string(I.num_vars) + "\t" + std::to_string(I.num_cons);
    for(size_t weight_index = 0; weight_index < I.w.size(); weight_index++) {
        for(size_t con_index = 0; con_index < I.U[weight_index].c.size(); con_index++) {
            if(I.U[weight_index].b.get(con_index) == 1) sign = -1;
            else sign = 1;
            weight = I.w[weight_index]*(double)sign;
            str.append("\n" + to_digits(weight, 4));
            for(size_t var_index = 0; var_index < I.U[weight_index].c[con_index].size(); var_index++) {
                if(I.U[weight_index].c[con_index].get(var_index) == 1) str.append("\t" + std::to_string(var_index));
            }
        }
    }
    return str;
}

bool save_instance(std::string filename, instance &I) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()) {
        notify("Unable to write to " + filename);
        return false;
    }
    outfile << to_string(I);
    outfile.close();
    return true;
}

//evaluate on 100 random bit strings and reject if value is not same
bool mc_equiv(instance &I1, instance &I2, std::mt19937_64 &eng) {
    if(I1.num_vars != I2.num_vars) return false;
    bitvector r(I1.num_vars);
    for(size_t trials = 0; trials < 100; trials++) {
        r.random(eng);
        if(error_frac(evaluate(I1, r), evaluate(I2, r)) > 1E-5) return false; //definitely not equivalent
    }
    return true; //might not be equivalent but we didn't find a difference
}

std::string con_string(const instance &I) {
    std::ostringstream out;
    for(size_t weight_index = 0; weight_index < I.w.size(); weight_index++) {
        if(weight_index > 0) out << "\n";
        out << to_digits(I.w[weight_index], 4);
        for(bitvector const &c : I.U[weight_index].c) out << "\n" << c;
    }
    return out.str();
}

std::string var_string(const instance &I) {
    std::ostringstream out;
    for(size_t weight_index = 0; weight_index < I.w.size(); weight_index++) {
        if(weight_index > 0) out << "\n";
        out << to_digits(I.w[weight_index], 4);
        for(bitvector const &v : I.U[weight_index].v) out << "\n" << v;
    }
    return out.str();
}

//Throughout this codebase our sign convention is that the value is the number of unsatisfied clauses minus
//the number of satisfied clauses. That is, this is a cost function to minimize.
int evaluate(unweighted_instance &UI, const bitvector &x) {
    size_t constraint_index;
    for(constraint_index = 0; constraint_index < UI.c.size(); constraint_index++) {
        UI.s.set(constraint_index, (UI.c[constraint_index] & x).count()%2);
    }
    UI.s += UI.b;
    return (int)UI.c.size()-2*(int)UI.s.count(); //+1 for each violated, -1 for each satisfied
}

//Throughout this codebase our sign convention is that the value is the number of unsatisfied clauses minus
//the number of satisfied clauses. That is, this is a cost function to minimize.
int diff(const unweighted_instance &UI, size_t var_index) {
    int before = UI.s.count(); //We could make (probably tiny) perf improvement by storing this
    bitvector new_s = UI.s + UI.v[var_index];
    int after = new_s.count();
    return (-2)*(after-before);
}

double evaluate(instance &I, const bitvector &x) {
    double val = 0.0;
    for(size_t weight_index = 0; weight_index < I.U.size(); weight_index++) {
        val += I.w[weight_index]*(double)evaluate(I.U[weight_index], x);
    }
    return val;
}

double diff(const instance &I, size_t var_index) {
    double delta = 0.0;
    for(size_t weight_index = 0; weight_index < I.U.size(); weight_index++) {
        delta += I.w[weight_index]*(double)diff(I.U[weight_index], var_index);
    }
    return delta;
}

void flip(unweighted_instance &UI, size_t var_index) {
    UI.s += UI.v[var_index]; //bitwise XOR
}

void flip(instance &I, size_t var_index) {
    for(size_t weight_index = 0; weight_index < I.U.size(); weight_index++) {
        I.U[weight_index].s += I.U[weight_index].v[var_index]; //bitwise XOR
    }
}

bool comp_t(event i, event j) {
    return ( i.t < j.t );
}

bool comp_tmag(event i, event j) {
    return ( i.tmag < j.tmag );
}

void magnitudes(const instance &I, std::vector <double> &M, std::vector <int> &s) {
    size_t i;
    double mag;
    M.clear();
    for(i = 0; i < I.w.size(); i++) {
        mag = fabs(I.w[i]);
        if(!approx_present(M, mag, 1.0E-5)) {
            M.push_back(mag);
        }
    }
    std::vector <int> local_s(I.U.size());
    for(i = 0; i < I.U.size(); i++) local_s[i] = I.U[i].c.size();
    s = local_s;
}

void plus_one(std::vector <int> &s) {
    for(unsigned int i = 0; i < s.size(); i++) s[i] += 1;
}

//the general case
void stats(std::vector <double> M, std::vector <int> s, int l, pstats &st) {
    if(M.size() != s.size()) { notify("Error: M.size() != s.size()"); return; }
    std::vector <int> f(s.size(),0);
    event e;
    std::vector <event> elist;
    double Z = 0.0;
    double max_logpZ = 0.0;
    std::vector <int> sbound = s;
    plus_one(sbound);
    //we need to be careful about numerical overflow
    for(vec_zero(f); vec_lessthan(f,sbound); vec_increment(f,sbound)) {
        double t = 0.0;
        for(unsigned int i = 0; i < s.size(); i++) t += M[i]*(double)(2*f[i]-s[i]);
        double lpz = 0.0;
        if(fabs(t) < 1.0E-20) lpz = -100.0; //if t = 0 log(t*t) = -inf, which causes trouble on some machines
        else {
            for(unsigned int i = 0; i < s.size(); i++) lpz += log_binomial(s[i],f[i]);
            lpz += log(t*t)*(double)l;
        }
        if(lpz > max_logpZ) max_logpZ = lpz;
    }
    for(vec_zero(f); vec_lessthan(f,sbound); vec_increment(f,sbound)) {
        double t = 0.0;
        for(unsigned int i = 0; i < s.size(); i++) t += M[i]*(double)(2*f[i]-s[i]);
        double lpz = 0.0;
        if(fabs(t) < 1.0E-20) lpz = -100.0; //if t = 0 log(t*t) = -inf, which causes trouble on some machines
        else {
            for(unsigned int i = 0; i < s.size(); i++) lpz += log_binomial(s[i],f[i]);
            lpz += log(t*t)*(double)l;
        }
        e.f = f;
        e.t = t;
        if(lpz - max_logpZ < -25) e.p = 0.0;
        else e.p = exp(lpz - max_logpZ); //normalization chosen to avoid overflow but otherwise arbitrary
        elist.push_back(e);
    }
    for(unsigned int i = 0; i < elist.size(); i++) Z += elist[i].p;
    for(unsigned int i = 0; i < elist.size(); i++) elist[i].p = elist[i].p/Z; //now we have the true probabilities
    for(unsigned int i = 0; i < elist.size(); i++) elist[i].tmag = fabs(elist[i].t);
    double psum = 0.0;
    for(unsigned int i = 0; i < elist.size(); i++) psum += elist[i].p;
    if(!approx_equal(psum,1.0)) notify("Error: psum != 1.0");
    st.mean_mag = 0.0;
    for(unsigned int i = 0; i < elist.size(); i++) st.mean_mag += elist[i].p*elist[i].tmag; //calculate expectation value
    double maxp = 0;
    double maxp_tmag = 0;
    for(unsigned int i = 0; i < elist.size(); i++) {
        if(elist[i].p > maxp) {
            maxp = elist[i].p;
            maxp_tmag = elist[i].tmag;
        }
    }
    st.max_p_tmag = maxp_tmag;
    std::sort(elist.begin(), elist.end(), comp_tmag); //sort by t to compute median
    unsigned int median_index;
    double p_cumulative = 0.0;
    unsigned int i = 0;
    do {
        p_cumulative += elist[i++].p;
    }while(p_cumulative < 0.5 && i < elist.size());
    if(i == elist.size()-1) notify("Error: p_cumulative=1/2 not found.");
    median_index = i-1;
    //std::cout << "p_cumulative = " << p_cumulative << std::endl;
    //std::cout << "median is entry " << median_index << " out of " << elist.size() << std::endl;
    st.median_mag = elist[median_index].tmag;
}

void print_pstats(const pstats &s) {
    std::cout << fws("mean_mag:") << fwf(s.mean_mag) << std::endl;
    std::cout << fws("median_mag:") << fwf(s.median_mag) << std::endl;
    std::cout << fws("max_p_tmag:") << fwf(s.max_p_tmag) << std::endl;
}

//all clauses satisfied means x * parity_check_matrix = target
bitvector target(const instance &I) {
    bitvector returnval = ~I.U[0].b;
    for(size_t i = 1; i < I.U.size(); i++) returnval.append(~I.U[i].b);
    return returnval;
}

typedef struct {
    double weight;
    size_t index;
}ws;

bool comp_weight(ws i, ws j) {
    return ( i.weight > j.weight );
}

bool is_priority_ordered(const instance &I) {
    for(size_t i = 1; i < I.w.size(); i++) if(I.w[i-1] < I.w[i]) return false;
    return true;
}

//order the instances by weight from highest to lowest
void priority_order(instance &I) {
    if(is_priority_ordered(I)) return;
    ws w;
    std::vector <ws> vw;
    for(size_t i = 0; i < I.w.size(); i++) {
        w.weight = I.w[i];
        w.index = i;
        vw.push_back(w);
    }
    std::sort(vw.begin(), vw.end(), comp_weight);
    std::vector <unweighted_instance>new_U;
    std::vector <double> new_w;
    for(const auto & wp : vw) {
        new_U.push_back(I.U[wp.index]);
        new_w.push_back(I.w[wp.index]);
    }
    I.U = new_U;
    I.w = new_w;
}

int gaussheur(const bitmatrix &H, const bitvector &t, bitvector &sol) {
    bitmatrix G, R;
    H.reduced_row_echelon_decomp(G, R);
    bitvector y(H.num_rows()); 
    size_t col_index = 0;
    size_t row_index = 0;
    while(row_index < H.num_rows() && col_index < H.num_cols()) {
        if(R.get(row_index, col_index)) {
            y.set(row_index, t.get(col_index));
            row_index++;
        }
        col_index++;
    }
    sol = y * G;
    int satisfied = (int)H.num_cols() - (int)(sol * H + t).count();
    return satisfied;
}

int descend(instance &I, const bitvector &init, bitvector &sol, bool verbose, bool backward) {
    bitvector x = init;
    double val = evaluate(I, x);
    int i,t;
    double delta;
    int accepted;
    bool accept;
    if(verbose) std::cout << fws("sweep") << fws("val") << fws("frac_acc") << std::endl;
    double best_val = val;
    bitvector best_x = x;
    int maxiter = 100000;
    double valminusten;
    int start = 0;
    int end = (int)I.num_vars-1;
    int increment = 1;
    if(backward) {
        start = (int)I.num_vars-1;
        end = 0;
        increment = -1;
    }
    for(t = 0; t < maxiter; t++) { //increment timestep
        accepted = 0;
        bool done = false;
        for(i = start; !done; i += increment) { //sweep through variable index
            accept = false;
            delta = diff(I, i);
            if(delta <= 0) accept = true;
            if(accept) {
                val += delta;
                flip(I, i);
                x.flip(i);
                accepted++;
            }
            if(val < best_val) {
                best_val = val;
                best_x = x;
            }
            if(i == end) done = true;
        }
        if(t%10 == 0 || accepted == 0) {
            if(verbose) std::cout << fwi(t+1) << fwf(val) << fwf((double)accepted/(double)(I.num_vars)) << std::endl;
            valminusten = val;
            if(t > 10) {
                if(valminusten == val) break; //no improvement in the last ten iterations
            }
        }
    }
    double final_val = evaluate(I,x);
    if(fabs(final_val - val) > 0.001) std::cout << "ERROR: faulty diffential evaluation!" << std::endl;
    int satisfied = (double(I.num_cons) - double(best_val)/double(I.w.at(0)))/2.0;
    if(verbose) {
        std::cout << "final x:" << std::endl;
        std::cout << x << std::endl;
        std::cout << "best x:" << std::endl;
        std::cout << best_x << std::endl;
        std::cout << fws("n") << fws("m") << fws("sweeps") << fws("val") << fws("best_val") << std::endl;
        std::cout << fwi(I.num_vars) << fwi(I.num_cons) << fwi(t) << fwf(evaluate(I,x)) << fwf(best_val) << std::endl;
        if(I.w.size() == 1) std::cout << satisfied << " clauses satisfied out of " << I.num_cons<<std::endl;
        else std::cout << "Unknown number of satisfied clauses due to different weights." << std::endl;
    }
    if(I.w.size() != 1) std::cout << "Warning: this is a weighted instance. The descent algorithm was not designed for this." << std::endl;
    sol = x;
    return satisfied;
}
