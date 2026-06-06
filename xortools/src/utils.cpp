#include <sstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include "utils.hpp"

//If the string hasn't been shown before, print it out. Otherwise do nothing.
void notify(std::string s, std::ostream &stream) {
    static std::vector <std::string> previous;
    static std::ostream *output = &std::cout;
    if(s == "REDIRECT") {
        output = &stream;
        return;
    }
    if(s == "CLEAR") {
        previous.clear();
        return;
    }
    std::vector<std::string>::iterator it;
    it = std::find(previous.begin(), previous.end(), s);
    if(it == previous.end()) { //it is a new message
        previous.push_back(s);
        (*output) << s << std::endl;
    }
}

bool contains(std::string s, char c) {
    if (s.find(c) != std::string::npos) return true;
    return false;
}

bool contains(std::vector<uint16_t>v, uint16_t x) {
    for(const auto & e : v) if(x == e) return true;
    return false;
}

std::vector <std::string> tokenize(const std::string &s, std::string separators) {
    std::vector <std::string> tokens;
    std::string token;
    for(size_t i = 0; i < s.size(); i++) {
        if(contains(separators, s[i])) { //s[i] is a separator character
            if(token.size() > 0) {
                tokens.push_back(token);
                token.clear();
            }
        }
        else token.push_back(s[i]);
    }
    if(token.size() > 0) tokens.push_back(token);
    return tokens;
}

std::vector <std::string> tokenize(const std::string &s, char separator) {
    std::vector <std::string> tokens;
    std::string token;
    for(size_t i = 0; i < s.size(); i++) {
        if(s[i] == separator) { //s[i] is a separator character
            if(token.size() > 0) {
                tokens.push_back(token);
                token.clear();
            }
        }
        else token.push_back(s[i]);
    }
    if(token.size() > 0) tokens.push_back(token);
    return tokens;
}

//pops the first token off the string and returns it
std::string poptoken(std::string &input, char separator) {
    std::string token;
    for(size_t i = 0; i < input.size(); i++) {
        if(input[i] == separator) {
            if(token.size() > 0) {
                while(input[i] == separator && i < input.size()) i++;
                input = input.substr(i,input.length()); //will stop at the end, actually
                return token;
            }
        }
        else token.push_back(input[i]);
    }
    input = "";
    return token;
}

//pops the first token off the string and returns it
std::string poptoken(std::string &input, std::string separators) {
    std::string token;
    for(size_t i = 0; i < input.size(); i++) {
        if(contains(separators, input[i])) {
            if(token.size() > 0) {
                while(contains(separators, input[i]) && i < input.size()) i++;
                input = input.substr(i,input.length()); //will stop at the end, actually
                return token;
            }
        }
        else token.push_back(input[i]);
    }
    input = "";
    return token;
}

bool approx_equal(double a, double b, double tolerance) {
    if(fabs(a-b) < tolerance) return true;
    return false;
}

bool approx_present(const std::vector <double> &list, double val, double tolerance) {
    for(double const &v : list) if(fabs(v-val) < tolerance) return true;
    return false;
}

size_t approx_find(const std::vector <double> &list, double val, double tolerance) {
    for(size_t i = 0; i < list.size(); i++) {
        double diff = fabs(list[i]-val);
        if(diff < tolerance) return i;
    }
    notify("Error: approximate value not found.");
    return 0;
}

double error_frac(double val, double approx) {
    if(val == approx) return 0.0;
    if(fabs(val) < 1E-20) {
        notify("Error: error_frac giving up due to fear of overflow.");
        return 0.0;
    }
    return fabs(approx-val)/fabs(val);
}

std::string to_digits(double value, int n) {
    std::ostringstream out;
    if(n < 1 || n > 100) {
        notify("Precision out of range in to_digits.");
        out << value;
        return out.str();
    }
    out.precision(n);
    out << std::fixed << value;
    return out.str();
}

int trailing_zeros(std::string s) {
    auto iter = s.end()-1;
    int zeros = 0;
    int points = 0;
    bool trailing = true;
    while(iter != s.begin() && points == 0) {
        switch(*iter) {
            case '0':
                if(trailing) zeros++;
                break;
            case '.':
                points++;
                break;
            default:
                trailing = false;
                break;
        }
        iter--;
    }
    if(points > 0) return zeros;
    return 0;
}

int afterdot(std::string s) {
    size_t val = 0;
    bool dotfound = false;
    for(int i = s.length()-1; i >= 0; i--) {
        if(s[i] == '.') {
            dotfound = true;
            break;
        }
        val++;
    }
    if(dotfound) return val;
    return 0;
}

std::string trim(std::string numstring) {
    int t = trailing_zeros(numstring);
    std::string trimmed = numstring;
    if(t > 3) {
        trimmed.erase(trimmed.end()-(t-3),trimmed.end());
    }
    return trimmed;
}

std::string trim_max(std::string numstring, size_t width) {
    int t = trailing_zeros(numstring);
    std::string trimmed = numstring;
    if(t > 3) {
        trimmed.erase(trimmed.end()-(t-3),trimmed.end());
    }
    if(trimmed.length() > width-1) {
        int to_remove = trimmed.length() - width + 1;
        int available = afterdot(trimmed)-1;
        if(available < to_remove) to_remove = available;
        trimmed.erase(trimmed.end()-to_remove,trimmed.end());
    }
    return trimmed;
}

std::string fwf(double num, size_t width) {
    std::string fullstring;
    if(fabs(num) > 1E5) {
        std::stringstream ss;
        ss << std::setprecision(3) << std::scientific << num;
        fullstring = ss.str();
    }
    else fullstring = trim_max(std::to_string(num), width);
    fullstring = fullstring + " "; //at least one trailing space even if it runs over width
    while(fullstring.length() < width) fullstring = fullstring + " "; //pad with spaces if under width
    return fullstring;
}

std::string fws(std::string s, size_t width) {
    std::string fullstring = s;
    while(fullstring.length() < width) fullstring = fullstring + " ";
    return fullstring;
}

std::string fwi(long int num, size_t width) {
    std::string fullstring = std::to_string(num);
    while(fullstring.length() < width) fullstring = fullstring + " ";
    return fullstring;
}

std::string vec_string(std::vector <int> v, std::string separator) {
    unsigned int i;
    std::string str;
    for(i = 0; i < v.size(); i++) {
        str.append(std::to_string(v[i]));
        if(i < v.size() - 1) str.append(separator);
    }
    return str;
}

std::string vec_string(std::vector <std::string> v, std::string separator) {
    unsigned int i;
    std::string str;
    for(i = 0; i < v.size(); i++) {
        str.append(v[i]);
        if(i < v.size() - 1) str.append(separator);
    }
    return str;
}

std::string vec_string(std::vector <size_t> v, std::string separator) {
    unsigned int i;
    std::string str;
    for(i = 0; i < v.size(); i++) {
        str.append(std::to_string(v[i]));
        if(i < v.size() - 1) str.append(separator);
    }
    return str;
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

std::string vec_string(std::vector <double> v, std::string separator) {
    unsigned int i;
    std::string str;
    for(i = 0; i < v.size(); i++) {
        str.append(to_string_with_precision(v[i], 3));
        if(i < v.size() - 1) str.append(separator);
    }
    return str;
}

double logfactorial(const int n) {
    int i;
    double val = 0.0;
    for(i = 2; i <= n; i++) val += log((double)i);
    return val;
}

double log_binomial(const int n, const int k) {
    return logfactorial(n) - logfactorial(k) - logfactorial(n-k);
}

double binomial(int n, int k) {
    if(k < 0 || k > n) return 0.0;
    if(k == 0 || k == n) return 1.0;
    double lb = log_binomial(n,k);
    if(fabs(lb) > 700) notify("Warning: likely overflow in binomial.");
    return exp(log_binomial(n, k));
}

void vec_zero(std::vector <int> &f) {
    fill(f.begin(), f.end(), 0);
}

bool vec_lessthan(std::vector <int> f, std::vector <int> s) {
    unsigned int i;
    if(f.size() != s.size()) {
        std::cout << "Error: range vector and iteration vector differ in size." << std::endl;
        return false;
    }
    for(i = 0; i < f.size(); i++) {
        if(f[i] >= s[i]) return false;
    }
    return true;
}

void vec_increment(std::vector <int> &f, std::vector <int> s) {
    unsigned int i;
    if(f.size() != s.size()) {
        std::cout << "Error: range vector and iteration vector differ in size." << std::endl;
        return;
    }
    for(i = 0; i < f.size(); i++) {
        if(f[i] < s[i]-1 || i == f.size() - 1) {
            f[i] = f[i] + 1;
            return;
        }
        //otherwise   
        f[i] = 0;
    }
}

//the binary entropy
double h2(double p) {
    if(p == 0.0) return 0.0;
    if(p == 1.0) return 0.0;
    return (p - 1.0)*log2(1.0-p) - p * log2(p);
}

void dispatcher(int numtasks, int numworkers, int worker_id, int &start, int &limit) {
    if(worker_id < 0 || worker_id >= numworkers) {
        std::cout << "Error!" << std::endl;
        start = 0;
        limit = 0;
    }
    double tasks_per_worker = (double)numtasks/(double)numworkers;
    double fstart = tasks_per_worker * (double)worker_id;
    double flimit = tasks_per_worker * (double)(worker_id+1);
    start = (int)std::round(fstart);
    limit = (int)std::round(flimit);
}

void replace_all(std::string& data, std::string to_search, std::string replace_str){
    size_t pos = data.find(to_search);
    while(pos != std::string::npos){
        data.replace(pos, to_search.size(), replace_str);
        pos = data.find(to_search, pos + replace_str.size());
    }
}

std::string mathematica_numberstring(double x) {
    std::ostringstream buffer;
    buffer << x;
    std::string returnval = buffer.str();
    replace_all(returnval, "e", "*10^");
    return returnval;
}

//factorial within limits of int. (n = 0 to 12)
int ifactorial(int n) {
    switch(n) {
        case 0:  return 1;
        case 1:  return 1;
        case 2:  return 2;
        case 3:  return 6;
        case 4:  return 24;
        case 5:  return 120;
        case 6:  return 720;
        case 7:  return 5040;
        case 8:  return 40320;
        case 9:  return 362880;
        case 10: return 3628800;
        case 11: return 39916800;
        case 12: return 479001600;
        default:
            notify("Error: n out of range in ifactorial.");
            return 1;
    }
}

//Constructor: initializes the first combination {0, 1, 2, ..., k-1}.
subset::subset(size_t n, size_t k) {
    if (k > n) {
        notify("Error: invalid parameters in call to subset constructor");
        n = 0;
        k = 0;
        return;
    }
    kval = k;
    nval = n;
    done = false;
    current.resize(kval);
    for (size_t i = 0; i < kval; i++) current[i] = i;
}

//Returns the current subset (combination) as a vector.
std::vector<size_t> subset::indices() const {
    return current;
}

//Returns true if there are more subsets to generate.
bool subset::notdone() const {
    return !done;
}

//Advances to the next combination in lexicographical order.
void subset::increment() {
    if (done) return;
    //Find the rightmost element that can be incremented.
    int i = static_cast<int>(kval) - 1;
    while (i >= 0 && current[i] == nval - kval + i) i--;
    if (i < 0) done = true;
    else {
        //Increment current element and adjust subsequent indices.
        (current[i])++;
        for (size_t j = i + 1; j < kval; ++j) current[j] = current[j - 1] + 1;
    }
}
