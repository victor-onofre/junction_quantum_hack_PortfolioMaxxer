#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "bitmatrix.hpp"
#include "utils.hpp"
#include "xorsat.hpp"
#include "ldpc.hpp"

//LP format doesn't use a * symbol
#define TIMES " "

typedef struct {
    double upper_bound;
    double lower_bound;
    std::string label;
} lpvar;

enum contype { LT, LE, GT, GE, EQ };

enum objtype { MIN, MAX };

//LP format uses =, not ==
const std::vector<std::string> constring = { "<", "<=", ">", ">=", "=" };

const double INF = std::numeric_limits<double>::infinity();

typedef struct {
    contype type;
    std::vector <size_t> indices;
    std::vector <double> coefficients;
    double RHS;
    std::string label;
} lpcon;

class sparse_lp;

class sparse_lp {
    public:
        //add_var and add_constraint return the index of the var or constraint
        int add_constraint(const std::vector<double> &coeffs, const std::vector<size_t> &indices, double rhs, contype type, std::string label= "");
        int add_constraint(const std::vector<double> &coeffs, const std::vector<std::string> &labels, double rhs, contype type, std::string label= "");
        //bounds can also be INF or -INF
        int add_var(double lower_bound, double upper_bound, std::string label= "");
        bool add_objective(const std::vector<double> &coeffs, const std::vector<size_t> &indices, objtype type);
        bool add_objective(const std::vector<double> &coeffs, const std::vector<std::string> &labels, objtype type);
        std::string obj_string() const;
        std::string var_string(size_t var_index) const;
        std::string var_string(std::string label) const;
        std::string con_string(size_t con_index) const;
        friend std::ostream& operator<<(std::ostream& os, const sparse_lp& lin); //stream e.g. to cout
        bool save_LP(std::string filename, std::string comments = "") const;
        bool save_MPS(std::string filename, std::string comments = "") const;
        size_t num_vars() const;
        size_t num_cons() const;
    private:
        std::vector <lpvar> vars;
        std::vector <lpcon> cons;
        std::vector <std::vector<size_t>> cols;      //the column indices as used in MPS format
        std::vector <std::vector<double>> colcoeffs; //the column coefficients as used in MPS format
        std::unordered_map<std::string, size_t> varmap;
        objtype objective_type;
        std::vector <size_t> objective_indices;
        std::vector <double> objective_coefficients;
};

size_t sparse_lp::num_vars() const {
    return vars.size();
}

size_t sparse_lp::num_cons() const {
    return cons.size();
}

int sparse_lp::add_constraint(const std::vector<double> &coeffs, const std::vector<size_t> &indices, double rhs, contype type, std::string label) {
    lpcon new_constraint;
    if(coeffs.size() != indices.size()) {
        std::cout << "Error: coefficient list does not have same length as index list." << std::endl;
        return -1;
    }
    for(const auto & i : indices) {
        if(i >= vars.size()) {
            std::cout << "Error: constraint references a variable that has not been created." << std::endl;
            return -1;
        }
    }
    if(label == "") label = "c" + std::to_string(cons.size());
    new_constraint.label = label;
    new_constraint.indices = indices;
    new_constraint.coefficients = coeffs;
    new_constraint.type = type;
    new_constraint.RHS = rhs;
    for(size_t i = 0; i < indices.size(); i++) {
        cols[indices[i]].push_back(cons.size());
        colcoeffs[indices[i]].push_back(coeffs[i]);
    }
    cons.push_back(new_constraint);
    return cons.size() - 1;
}

int sparse_lp::add_constraint(const std::vector<double> &coeffs, const std::vector<std::string> &labels, double rhs, contype type, std::string label) {
    if(coeffs.size() != labels.size()) {
        std::cout << "Error: coefficient list does not have same length as label list." << std::endl;
        return -1;
    }
    std::vector<size_t> indices(labels.size());
    std::unordered_map<std::string, size_t>::iterator itr; 
    for(size_t i = 0; i < labels.size(); i++) {
        itr = varmap.find(labels[i]);
        if(itr == varmap.end()) {
            std::cout << "Error: variable labeled " << itr->first << " not found." << std::endl;
            return -1;
        }
        indices[i] = itr->second;
    }
    return add_constraint(coeffs, indices, rhs, type, label);
}

bool sparse_lp::add_objective(const std::vector<double> &coeffs, const std::vector<size_t> &indices, objtype type) {
    if(coeffs.size() != indices.size()) {
        std::cout << "Error: coefficient list does not have same length as index list." << std::endl;
        return false;
    }
    for(const auto & i : indices) {
        if(i >= vars.size()) {
            std::cout << "Error: constraint references a variable that has not been created." << std::endl;
            return false;
        }
    }
    objective_type = type;
    objective_coefficients = coeffs;
    objective_indices = indices;
    return true;
}

bool sparse_lp::add_objective(const std::vector<double> &coeffs, const std::vector<std::string> &labels, objtype type) {
    if(coeffs.size() != labels.size()) {
        std::cout << "Error: coefficient list does not have same length as label list." << std::endl;
        return false;
    }
    std::vector<size_t> indices(labels.size());
    std::unordered_map<std::string, size_t>::iterator itr; 
    for(size_t i = 0; i < labels.size(); i++) {
        itr = varmap.find(labels[i]);
        if(itr == varmap.end()) {
            std::cout << "Error: variable labeled " << itr->first << " not found." << std::endl;
            return false;
        }
        indices[i] = itr->second;
    }
    return add_objective(coeffs, indices, type);
}

int sparse_lp::add_var(double lower_bound, double upper_bound, std::string label) {
    if(lower_bound > upper_bound) {
        std::cout << "Error: variable lower bound is above upper bound." << std::endl;
        return -1;
    }
    if(label == "") label = "x" + std::to_string(vars.size());
    lpvar new_var;
    new_var.lower_bound = lower_bound;
    new_var.upper_bound = upper_bound;
    new_var.label = label;
    vars.push_back(new_var);
    std::vector<size_t> empty;
    cols.push_back(empty);
    std::vector<double> emptyd;
    colcoeffs.push_back(emptyd);
    varmap[label] = vars.size()-1;
    return vars.size() - 1;
}

std::string sparse_lp::var_string(size_t var_index) const {
    std::string str;
    if(var_index >= vars.size()) {
        std::cout << "Error: var_index out of bounds in print_var." << std::endl;
        return str;
    }
    const lpvar & var = vars[var_index];
    if(var.lower_bound == -INF) str += "-Inf <= ";
    else str += std::to_string(var.lower_bound) + " <= ";
    str += var.label;
    if(var.upper_bound == INF) str += " <= Inf";
    else str += " <= " + std::to_string(var.upper_bound);
    return str;
}

std::string sparse_lp::var_string(std::string label) const {
    const auto & itr = varmap.find(label);
    if(itr == varmap.end()) {
        std::cout << "Error: variable labeled " << itr->first << " not found." << std::endl;
        return "";
    }
    return var_string(itr->second);
}

std::string termstring(double coeff, std::string label, bool leading) {
    if(coeff == 1.0) {
        if(leading) return label;
        else return " + " + label;
    }
    if(coeff == -1.0) return " - " + label;
    if(coeff < 0.0) {
        if(leading) return std::to_string(coeff) + TIMES + label;
        return " - " + std::to_string(fabs(coeff)) + TIMES + label;
    }
    if(coeff > 0.0) {
        if(leading) return std::to_string(coeff) + TIMES + label;
        return " + " + std::to_string(coeff) + TIMES + label;
    }
    return ""; //coeff == 0.0
}

std::string sparse_lp::con_string(size_t con_index) const {
    std::string str;
    if(con_index >= cons.size()) {
        std::cout << "Error: con_index out of bounds in print_con." << std::endl;
        return str;
    }
    const lpcon & con = cons[con_index];
    str += con.label + ": ";
    for(size_t i = 0; i < con.indices.size(); i++) str += termstring(con.coefficients[i], vars[con.indices[i]].label, !i);
    str += " " + constring[con.type] + " " + std::to_string(con.RHS);
    return str;
}

std::string sparse_lp::obj_string() const {
    std::string str;
    if(objective_type == MAX) str = "Maximize\n\t";
    else str = "Minimize\n\t";
    for(size_t i = 0; i < objective_indices.size(); i++) str += termstring(objective_coefficients[i], vars[objective_indices[i]].label, !i);
    return str;
}

std::ostream& operator<<(std::ostream& os, const sparse_lp& lin) {
    os << lin.obj_string() << '\n';
    os << "Subject To\n";
    for(size_t j = 0; j < lin.num_cons(); j++) os << '\t' << lin.con_string(j) << '\n';
    os << "Bounds\n";
    for(size_t i = 0; i < lin.num_vars(); i++) os << '\t' << lin.var_string(i) << '\n';
    os << "End" << std::endl;
    return os;
}

sparse_lp feldman_lp(const bitmatrix &H, const bitvector &c, double p_flip) {
    sparse_lp lin;
    size_t cons = H.num_rows();
    size_t vars = H.num_cols();
    for(size_t i = 0; i < vars; i++) lin.add_var(0.0,1.0);
    for(size_t j = 0; j < cons; j++) {
        bitvector convec = H.row_vec(j);
        std::vector<size_t> varlist = convec.to_sparse();
        std::vector<size_t> empty;
        std::vector<std::vector<size_t>> var_omegalist(varlist.size(), empty);
        int D = varlist.size();
        if(D > 20) {
            std::cout << "Error: D is too big" << std::endl;
            return lin;
        }
        int omegamax = 1<<D;
        bitvector omegastring;
        std::vector<size_t>omegalist;
        for(int i = 0; i < omegamax; i++) {
            omegastring.from_num(i,D);
            if(!(omegastring.count() % 2)) {
                std::string label = "omega_" + std::to_string(j) + "_" + omegastring.to_str();
                size_t omega_index = lin.add_var(0.0,1.0,label);
                std::vector<size_t> omega_sparse = omegastring.to_sparse();
                for(const auto &index : omega_sparse) var_omegalist[index].push_back(omega_index);
                omegalist.push_back(omega_index);
            }
        }
        std::vector ones(omegalist.size(), 1.0);
        lin.add_constraint(ones, omegalist, 1.0, EQ); //sum of omegas from this constraint equals 1
        //add the f sum constaints
        for(size_t i = 0; i < varlist.size(); i++) {
            size_t var_index = varlist[i];
            std::vector<size_t> indexlist = var_omegalist[i];
            indexlist.push_back(var_index);
            std::vector<double> coeffs(indexlist.size(),1.0);
            coeffs[coeffs.size()-1] = -1.0;
            lin.add_constraint(coeffs, indexlist, 0.0, EQ);
        }
    }
    double gamma = log(p_flip/(1.0-p_flip));
    std::vector<double> obj_coeffs(c.size());
    std::vector<size_t> obj_indices(c.size());
    for(size_t i = 0; i < c.size(); i++) {
        obj_indices[i] = i;
        if(c.get(i)) obj_coeffs[i] = gamma;
        else obj_coeffs[i] = -gamma;
    }
    lin.add_objective(obj_coeffs, obj_indices, MIN);
    return lin;
}

bool sparse_lp::save_LP(std::string filename, std::string comments) const {
    std::ofstream outfile(filename);
    if(!outfile.is_open()) {
        notify("Unable to write to " + filename);
        return false;
    }
    if(!comments.empty()) {
        std::vector<std::string> lines = tokenize(comments, '\n');
        for(const auto & line : lines) outfile << "\\ " << line << '\n';
        outfile << '\n';
    }
    outfile << *this;
    outfile.close();
    return true;
}

bool sparse_lp::save_MPS(std::string filename, std::string comments) const {
    const std::string sp = "    ";
    std::ofstream outfile(filename);
    if(!outfile.is_open()) {
        notify("Unable to write to " + filename);
        return false;
    }
    std::vector <std::vector<size_t>> local_cols = cols;
    std::vector <std::vector<double>> local_colcoeffs = colcoeffs;
    //paste on the objective function as if it were an additional constraint----------
    for(size_t i = 0; i < objective_indices.size(); i++) {
        local_cols[objective_indices[i]].push_back(cons.size());
        local_colcoeffs[objective_indices[i]].push_back(objective_coefficients[i]);
    }
    std::vector <lpcon> local_cons = cons;
    lpcon objcon;
    objcon.label = "OBJ";
    objcon.coefficients = objective_coefficients;
    objcon.indices = objective_indices;
    local_cons.push_back(objcon);
    //--------------------------------------------------------------------------------
    if(!comments.empty()) {
        std::vector<std::string> lines = tokenize(comments, '\n');
        for(const auto & line : lines) outfile << "* " << line << '\n';
        outfile << '\n';
    }
    size_t lastindex = filename.find_last_of("."); 
    std::string name = filename.substr(0, lastindex);
    outfile << "NAME" << sp << name << '\n';
    if(objective_type == MAX) outfile << "OBJSENSE\n\tMAX\n";
    else outfile << "OBJSENSE\n" << sp << "MIN\n";
    outfile << "ROWS\n";
    for(const auto & con : cons) {
        if(con.type == EQ) outfile << sp << "E" << sp;
        if(con.type == LT || con.type == LE) outfile << sp << "L" << sp;
        if(con.type == GT || con.type == GE) outfile << sp << "G" << sp;
        outfile << con.label << '\n';
    }
    outfile << sp << "N" << sp << "OBJ\n";
    outfile << "COLUMNS\n";
    //if the column has more than two entries we need to break it up
    for(size_t i = 0; i < local_cols.size(); i++) {
        size_t start = 0;
        while(start < local_cols[i].size()) {
            outfile << sp << vars[i].label;
            size_t eaten = 0;
            for(size_t j = start; j < local_cols[i].size() && j < start + 2; j++) {
                outfile << sp << local_cons[local_cols[i][j]].label << sp << std::to_string(local_colcoeffs[i][j]);
                eaten++;
            }
            start += eaten;
            outfile << '\n';
        }
    }
    outfile << "RHS\n";
    for(size_t i = 0; i < cons.size(); i++) outfile << sp << "RHS" << sp << cons[i].label << sp << std::to_string(cons[i].RHS) << '\n';
    outfile << "BOUNDS\n";
    for(size_t i = 0; i < vars.size(); i++) {
        if(vars[i].lower_bound == -INF) {
            if(vars[i].upper_bound == INF) outfile << sp << "FR" << sp << "BND" << sp << vars[i].label << '\n';
            else {
                outfile << sp << "MI" << sp << "BND" << sp << vars[i].label << '\n';
                outfile << sp << "UP" << sp << "BND" << sp << vars[i].label << sp << std::to_string(vars[i].upper_bound) << '\n';
            }
        }
        else {
            outfile << sp << "LO" << sp << "BND" << sp << vars[i].label << sp << std::to_string(vars[i].lower_bound) << '\n';
            if(vars[i].upper_bound != INF) outfile << sp << "UP" << sp << "BND" << sp << vars[i].label << sp << std::to_string(vars[i].upper_bound) << '\n';
        }
    }
    outfile << "ENDATA" << std::endl;
    outfile.close();
    return true;
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
    comments += "Random codeword:    " + random_codeword.to_str() + '\n';
    bitvector corrupted_codeword = apply_rand_err(random_codeword, eng, errors);
    comments += "Corrupted codeword: " + corrupted_codeword.to_str() + '\n';
    comments += "Errors:             " + (random_codeword + corrupted_codeword).to_str() + '\n';
    std::cout << comments;
    sparse_lp lin = feldman_lp(H, corrupted_codeword, p_flip);
    std::string filename = argv[1];
    size_t lastindex = filename.find_last_of("."); 
    filename = filename.substr(0, lastindex);
    filename += "_" + std::to_string(errors);
    std::string mpsname = filename + ".mps";
    filename += ".lp";
    lin.save_LP(filename, comments);
    lin.save_MPS(mpsname, comments);
    return 0;
}
