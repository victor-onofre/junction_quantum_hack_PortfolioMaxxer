#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <set>
#include <queue>

//If the string hasn't been shown before, print it out. Otherwise do nothing.
void notify(std::string s, std::ostream &stream = std::cout);

template <typename T>
std::vector <T> random_subset(size_t n, size_t k, std::mt19937_64 &eng, size_t min=0) {
    if(n < k + min) {
        notify("Error: n < k + min in random_subset");
        std::vector <T> zeros(n,0);
        return zeros;
    }
    std::vector <T> all(n-min);
    std::vector <T> returnval;
    for(T i = 0; i < all.size(); i++) all[i] = i+min;
    std::sample(all.begin(), all.end(), back_inserter(returnval), k, eng);
    return returnval;
}

std::vector <std::string> tokenize(const std::string &s, std::string separators);

std::vector <std::string> tokenize(const std::string &s, char separator);

//pops the first token off the string and returns it
std::string poptoken(std::string &input, char separator);

//pops the first token off the string and returns it
std::string poptoken(std::string &input, std::string separators);

bool approx_equal(double a, double b, double tolerance = 1E-5);

bool approx_present(const std::vector <double> &list, double val, double tolerance);

size_t approx_find(const std::vector <double> &list, double val, double tolerance);

std::string to_digits(double value, int n);

int trailing_zeros(std::string s);

std::string trim(std::string numstring);

double error_frac(double val, double approx); //In other words, percent error times 100.

//smart fixed-width output:
std::string fwf(double num, size_t width = 12);

std::string fws(std::string s, size_t width = 12);

std::string fwi(long int num, size_t width = 12);

std::string vec_string(std::vector <int> v, std::string separator);

std::string vec_string(std::vector <std::string> v, std::string separator);

std::string vec_string(std::vector <size_t> v, std::string separator);

std::string vec_string(std::vector <double> v, std::string separator);

double logfactorial(const int n);

double log_binomial(const int n, const int k);

double binomial(int n, int k);

void vec_zero(std::vector <int> &f);

bool vec_lessthan(std::vector <int> f, std::vector <int> s);

void vec_increment(std::vector <int> &f, std::vector <int> s);

//the binary entropy
double h2(double p);

bool contains(std::string s, char c);

bool contains(std::vector<uint16_t>v, uint16_t x);

//for simple multithreading workflows
void dispatcher(int numtasks, int numworkers, int worker_id, int &start, int &limit);

//replace all instances of to_search with replace_str
void replace_all(std::string& data, std::string to_search, std::string replace_str);

std::string mathematica_numberstring(double x); //mathematica scientific format using *10^ instead of e

template <typename T>
inline std::vector<size_t> argsort(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
bool all_distinct(const std::vector<T>& vec) {
  std::unordered_set<T> elementSet(vec.begin(), vec.end());
  return elementSet.size() == vec.size();
}

template <typename T>
std::string queue_string(const std::queue<T> &q, std::string separator = ",") {
    std::string returnval;
    std::queue<T> tempQueue = q; // Create a copy of the queue
    while (!tempQueue.empty()) {
      returnval +=  std::to_string(tempQueue.front()) + separator;
      tempQueue.pop();
    }
    return returnval;
}

template <typename T>
std::string set_string(const std::set<T> &s, std::string separator = ",") {
    std::string returnval;
    for (auto it = s.begin(); it != s.end(); ++it) {
        returnval += std::to_string(*it);
        if(std::next(it) != s.end()) returnval += separator;
    }
    return returnval;
}

template <typename T>
std::string multiset_string(const std::multiset<T> &s, std::string separator = ",") {
    std::string returnval;
    for (auto it = s.begin(); it != s.end(); ++it) {
        returnval += std::to_string(*it);
        if(std::next(it) != s.end()) returnval += separator;
    }
    return returnval;
}

//Iterate through subsets of size k like this: for(subset s(n,k); s.notdone(); s.increment())
class subset {
  public:
      subset(size_t n, size_t k);
      void increment();
      bool notdone() const;
      std::vector<size_t> indices() const;
  private:
      size_t nval;                 //Total number of elements.
      size_t kval;                 //Number of elements per subset.
      std::vector<size_t> current; //Current combination of indices.
      bool done;                   //Flag indicating if all subsets have been generated.
};

//factorial within limits of int. (n = 0 to 12)
int ifactorial(int n);

#endif
