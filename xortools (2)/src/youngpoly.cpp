#include <iostream>
#include <Eigen/Dense>
#include "polytools.hpp"

int main(int argc, char *argv[]) {
    if(argc != 7) {
        std::cout << "Usage: youngpoly m l tmin tmax listsize modulus" << std::endl;
        return 0;
    }
    int m = std::stoi(argv[1]);
    int l = std::stoi(argv[2]);
    int tmin = std::stoi(argv[3]);
    int tmax = std::stoi(argv[4]);
    int listsize = std::stoi(argv[5]);
    int modulus = std::stoi(argv[6]);
    std::cout << "Generating matrices" << std::endl;
    Eigen::MatrixXd A = generate_A(m, l, listsize, modulus);
    //std::cout << "AAT:\n" << A*A.transpose() << std::endl;
    std::cout << "t\tp_success" << std::endl;
    for(int t = tmin; t <= tmax; t++) {
        Eigen::MatrixXd Agt = A.rightCols(m-t+1);
        Eigen::MatrixXd AAgt = Agt*Agt.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AAgt);
        std::cout << t << "\t" << es.eigenvalues().row(l) << std::endl;
    }
    return 0;
}
