#include <string>
#include <random>
#include <thread>
#include <fstream>
#include "xoropt.hpp"
#include "bitmatrix.hpp"
#include "utils.hpp"

bool xorsat_instance::load(std::string filename) {
    std::vector <std::vector <std::string> > token_array;
    std::vector <std::string> firstline_tokens;
    std::vector <std::string> line_tokens;
    std::string line;
    int num_vars, num_cons;
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
    BT.zeros(num_vars,num_cons);
    v.zeros(num_cons);
    for(size_t con_index = 0; con_index < token_array.size(); con_index++) {
        double weight = std::stod(token_array[con_index][0]);
        if(fabs(weight) != 0.5) {
            notify("Weights other than +-1/2 are not supported.");
            return false;
        }
        if(weight > 0) v.set(con_index,1);
        for(size_t token_index = 1; token_index < token_array[con_index].size(); token_index++) {
            int one_index = std::stoi(token_array[con_index][token_index]);
            if(one_index < 0 || one_index > num_vars) {
                notify("Index out of range in " + filename);
                return false;
            }
            BT.set(one_index, con_index, 1);
        }
    }
    return true;
}

size_t xorsat_instance::num_vars() const {
    return BT.num_rows();
}

size_t xorsat_instance::num_cons() const {
    return BT.num_cols();
}

std::string xorsat_instance::to_string() {
    double weight;
    std::string str = std::to_string(num_vars()) + "\t" + std::to_string(num_cons());
    for(size_t con_index = 0; con_index < num_cons(); con_index++) {
        if(v.get(con_index) == 1) weight = -0.5;
        else weight = 0.5;
        str.append("\n" + to_digits(weight, 4));
        for(size_t var_index = 0; var_index < num_vars(); var_index++) {
            if(BT.get(var_index, con_index)) str.append("\t" + std::to_string(var_index));
        }
    }
    return str;
}

bool xorsat_instance::save(std::string filename) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()) {
        notify("Unable to write to " + filename);
        return false;
    }
    outfile << to_string();
    outfile.close();
    return true;
}

walker::walker(const xorsat_instance &instance) {
    evaluated = false;
    proposal_queued = false;
    I = &instance;
    x.zeros(I->num_vars());
    y.zeros(I->num_cons());
    s.zeros(I->num_cons());
}

int clauses_violated(const xorsat_instance &I, const bitvector &x) {
    bitvector s = x * I.BT + I.v;
    return s.count();
}

int walker::value() {
    if(evaluated) return val;
    //otherwise:
    //std::cout << "Evaluating from scratch." << std::endl;
    y = x * I->BT;
    s = y + I->v;
    val = s.count();
    evaluated = true;
    return val;
}

void walker::randomize(std::mt19937_64 &eng) {
    x.random(eng);
    evaluated = false;
    proposal_queued = false;
}

int walker::propose(size_t index) {
    if(index > I->num_vars()) {
        notify("Invalid index in move proposal");
        return 0;
    }
    if(!evaluated) {
        notify("Evaluating from scratch in move proposal.");
        value();
    }
    proposal_queued = true;
    flip_proposed = index;
    size_t new_w = new_weight(I->BT, s, index);
    diff_proposed = new_w - val;
    return diff_proposed;
}

void walker::accept() {
    if(!evaluated) {
        notify("Cannot accept move on unevaluated walker.");
        return;
    }
    if(!proposal_queued) {
        notify("Cannot accept proposal as none is queued");
        return;
    }
    x.flip(flip_proposed);
    s += I->BT.row_vec(flip_proposed);
    val += diff_proposed;
    proposal_queued = false;
}

void walker::reject() {
    proposal_queued = false;
}

void walker::set_x(const bitvector &xvals) {
    x = xvals;
    evaluated = false;
    proposal_queued = false;
}

bitvector walker::get_x() const{
    return x;
}

size_t walker::num_vars() const {
    return I->BT.num_rows();
}

size_t walker::num_cons() const {
    return I->BT.num_cols();
}

//routines to sort bitmatrix columns by Hamming weight-------------------------------------------------------------------

typedef struct {
    size_t index;
    size_t weight;
}colstats;

//we want descending order
bool greater_colstats(colstats i, colstats j) {
    return ( i.weight > j.weight );
}

//we want descending order
bool lesser_colstats(colstats i, colstats j) {
    return ( i.weight < j.weight );
}

std::vector<size_t> sorting_permutation(const bitmatrix &H, bool ascending) {
    std::vector<colstats> s(H.num_cols());
    for(size_t i = 0; i < H.num_cols(); i++) {
        s[i].index = i;
        s[i].weight = H.col_vec(i).count();
    }
    if(ascending) std::sort(s.begin(), s.end(), lesser_colstats);
    else std::sort(s.begin(), s.end(), greater_colstats);
    std::vector<size_t> returnval(H.num_cols());
    for(size_t i = 0; i < H.num_cols(); i++) returnval[i] = s[i].index;
    return returnval;
}

bool is_sorted(const bitmatrix &H, bool ascending) {
    size_t last_weight = H.col_vec(0).count();
    size_t this_weight;
    for(size_t i = 1; i < H.num_cols(); i++) {
        this_weight = H.col_vec(i).count();
        if(ascending && (this_weight < last_weight)) return false;
        if(!ascending && (this_weight > last_weight)) return false;
        last_weight = this_weight;
    }
    return true;
}

//optimization algorithms------------------------------------------------------------------------------------------------

//Warning: does not automatically randomize the initial walker position. If you want that do w.randomize() first.
void anneal(walker &w, int iter, double minbeta, double maxbeta, std::mt19937_64 &eng, bool verbose) {
    std::uniform_int_distribution<uint64_t> distr;
    int best_val = w.value();
    bitvector best_x = w.get_x();
    int t;
    size_t i;
    int delta;
    double p_accept;
    uint64_t threshold;
    bool accept;
    int accepted = 0;
    int rejected = 0;
    if(verbose) std::cout << fws("sweep") << fws("beta") << fws("val") << fws("frac_acc") << std::endl;
    double beta = minbeta;
    double delta_beta = (maxbeta - minbeta)/(double)iter;
    for(t = 0; t < iter; t++) { //increment timestep
        beta += delta_beta;
        for(i = 0; i < w.num_vars(); i++) { //sweep through variable index
            accept = false;
            delta = w.propose(i);
            if(delta <= 0) accept = true;
            if(!accept) {
                p_accept = exp(-1.0*beta*(double)delta);
                threshold = (uint64_t)(p_accept*(double)UINT64_MAX);
                if(distr(eng) < threshold) accept = true;
            }
            if(accept) {
                w.accept();
                accepted++;
            }
            else {
                w.reject();
                rejected++;
            }
            if(w.value() < best_val) {
                best_val = w.value();
                best_x = w.get_x();
            }
        }
        if(verbose && t%10 == 0) {
            std::cout << fwi(t+1) << fwf(beta) << fwi(w.value()) << fwf((double)accepted/(double)(accepted+rejected)) << std::endl;
            accepted = 0;
            rejected = 0;
        }
    }
    int final_val = w.value();
    w.set_x(w.get_x());
    if(final_val != w.value()) {
        std::cout << "ERROR: faulty diffential evaluation!" << std::endl;
    }
    if(verbose) {
        std::cout << "final x:" << std::endl;
        std::cout << w.get_x() << std::endl;
        std::cout << "best x:" << std::endl;
        std::cout << best_x << std::endl;
        std::cout << fws("n") << fws("m") << fws("sweeps") << fws("val") << fws("best_val") << std::endl;
        std::cout << fwi(w.num_vars()) << fwi(w.num_cons()) << fwi(t) << fwi(w.value()) << fwi(best_val) << std::endl;
        std::cout << w.num_cons() - w.value() << " clauses satisfied out of " << w.num_cons() << std::endl;
    }
}

//This is a greedy algorithm.
void descend(walker &w, bool verbose) {
    int best_val = w.value();
    int t, delta, valminusten, accepted;
    unsigned int i;
    int maxiter = 100000;
    if(verbose) std::cout << fws("sweep") << fws("val") << fws("frac_acc") << std::endl;
    for(t = 0; t < maxiter; t++) { //increment timestep
        accepted = 0;
        for(i = 0; i < w.num_vars(); i++) { //sweep through variable index
            delta = w.propose(i);
            if(delta <= 0) w.accept();
            else w.reject();
            if(w.value() < best_val) best_val = w.value();
        }
        if(t%10 == 0 || accepted == 0) {
            if(verbose) std::cout << fwi(t+1) << fwi(w.value()) << fwf((double)accepted/(double)(w.num_vars())) << std::endl;
            valminusten = w.value();
            if(t > 10) {
                if(valminusten == w.value()) break; //no improvement in the last ten iterations
            }
        }
    }
    int final_val = w.value();
    w.set_x(w.get_x());
    if(final_val != w.value()) std::cout << "Faulty differential evaluation!" << std::endl;
    if(verbose) {
        std::cout << "final x:" << std::endl;
        std::cout << w.get_x() << std::endl;
        std::cout << fws("n") << fws("m") << fws("sweeps") << fws("val") << fws("best_val") << std::endl;
        std::cout << fwi(w.num_vars()) << fwi(w.num_cons()) << fwi(t) << fwi(w.value()) << fwi(best_val) << std::endl;
        std::cout << w.num_cons() - w.value() << " clauses satisfied out of " << w.num_cons() << std::endl; 
    }
}

int truncation_heuristic(const xorsat_instance &I, bitvector &sol, bool verbose) {
    bitmatrix G, R;
    I.BT.reduced_row_echelon_decomp(G, R);
    bitvector y(I.BT.num_rows()); 
    size_t col_index = 0;
    size_t row_index = 0;
    while(row_index < I.BT.num_rows() && col_index < I.BT.num_cols()) {
        if(R.get(row_index, col_index)) {
            y.set(row_index, I.v.get(col_index));
            row_index++;
        }
        col_index++;
    }
    sol = y * G;
    int violated = (sol * I.BT + I.v).count();
    if(verbose) {
        std::cout << I.num_cons() - violated << " clauses satisfied out of " << I.num_cons() << std::endl;
        std::cout << "solution:\n" << sol << std::endl;
    }
    return violated;
}

void sort_instance(xorsat_instance &I, bool ascending, bool verbose) {
    if(is_sorted(I.BT, ascending)) {
        if(verbose) {
            if(ascending) std::cout << "Instance is already ascending" << std::endl;
            else std::cout << "Instance is already descending" << std::endl;
        }
        return;
    }
    std::vector<size_t> permutation = sorting_permutation(I.BT, ascending);
    I.BT = permute_columns(I.BT, permutation);
    I.v = permute_bits(I.v, permutation);
    if(!is_sorted(I.BT, ascending)) std::cout << "Error: H is not sorted" << std::endl;
}

int sort_truncate_descend(const xorsat_instance &I, bitvector &sol, bool reverse, bool verbose) {
    xorsat_instance IS = I;
    sort_instance(IS, reverse, verbose);
    truncation_heuristic(IS, sol, verbose);
    walker w(IS);
    w.set_x(sol);
    descend(w, verbose);
    sol = w.get_x();
    return w.value();
}

bool transfer_heaviest(xorsat_instance &from, xorsat_instance &to, size_t threshold) {
    bitmatrix B = from.BT.transpose();
    bitmatrix Bheavy, Blight;
    bitvector vheavy, vlight;
    bitvector row;
    for(size_t row_index = 0; row_index < B.num_rows(); row_index++) {
        row = B.row_vec(row_index);
        size_t weight = row.count();
        if(weight > threshold) {
            Bheavy.append_below(row);
            vheavy.push_back(from.v.get(row_index));
        }
        else {
            Blight.append_below(row);
            vlight.push_back(from.v.get(row_index));
        }
    }
    if(Bheavy.num_rows() == 0 || Blight.num_rows() == 0) {
        notify("Error: transfer_heaviest results in empty instance.");
        return false;
    }
    from.BT = Blight.transpose();
    from.v = vlight;
    to.BT = Bheavy.transpose();
    to.v = vheavy;
    return true;
}

void merge(const xorsat_instance &from, xorsat_instance &to) {
    to.v.append(from.v);
    to.BT.append_right(from.BT);
}

//Sergei Isakov's algorithm
int partial_anneal(const xorsat_instance &I, size_t weight_cutoff, bitvector &sol, uint64_t seed) {
    std::mt19937_64 eng(seed);
    xorsat_instance Iheavy, Ilight;
    Ilight = I;
    transfer_heaviest(Ilight, Iheavy, weight_cutoff);
    walker w1(Ilight);
    w1.randomize(eng);
    anneal(w1, 1000000, 0.5, 4.0, eng, false);
    bitvector x = w1.get_x();
    walker w2(I);
    w2.set_x(x);
    int violated_before_descent = w2.value();
    descend(w2, false);
    sol = w2.get_x();
    int violated_after_descent = w2.value();
    std::cout << violated_before_descent << " -> " << violated_after_descent << std::endl;
    return w2.value();
}

