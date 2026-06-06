#include <iostream>
#include "listsat.hpp"
#include "pmatrix.hpp"

int main(int argc, char *argv[]) {
    lsinstance I;
    pvector x;
    double val;
    if(argc != 3) {
        std::cout << "Usage: evalist instance.csv solution.psol" << std::endl;
        return 0;
    }
    if(!load_lsinstance(argv[1], I)) return 0;
    std::cout << "Loaded instance " << argv[1] << " with " << I.M.num_rows() << " variables and " << I.M.num_cols() << " clauses." << std::endl;
    if(!x.load_dense(argv[2])) return 0;
    std::cout << "Loaded solution " << argv[2] << " with " << x.size() << " variables." << std::endl << std::endl;
    //val is the number of unsatisfied clauses
    val = evaluate(I, x);
    std::cout << "number of unsatisfied clauses: " << val << std::endl;
    std::cout << "number of satisfied clauses: " << I.M.num_cols() - val << std::endl;
    std::cout << "(SAT - UNSAT): " << I.M.num_cols() - 2*val << std::endl;
    std::cout << "(1/2)*(SAT - UNSAT): " << 0.5*(I.M.num_cols() - 2.0*val) << std::endl;
    return 0;
}