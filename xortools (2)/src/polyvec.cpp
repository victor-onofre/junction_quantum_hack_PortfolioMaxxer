//compile with: g++ -Wall -O3 --std=c++11 youngpoly.cpp -o youngpoly -lntl -lgmp -pthread
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "polytools.hpp"

int main(int argc, char *argv[]) {
    if(argc != 3) {
        std::cout << "Usage: polyvec m l" << std::endl;
        return 0;
    }
    int m = std::stoi(argv[1]);
    int l = std::stoi(argv[2]);
    int listsize = 1;
    int modulus = 2;
    //std::cout << "Generating matrices" << std::endl;
    //std::cout << "A*A^T--------------------------" << std::endl;
    Eigen::MatrixXd A = generate_A(m, l, listsize, modulus);
    //std::cout << A*A.transpose() << std::endl;
    Eigen::MatrixXd D(m+1,m+1);
    for(int m_index = 0; m_index <= m; m_index++) D(m_index,m_index) = (double)m_index;
    Eigen::MatrixXd Q = A * D * A.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
    std::cout << m << '\t' << l << '\t' << es.eigenvalues().row(l) << std::endl;
    std::cout << "eigenvector---------------------------" << std::endl;
    Eigen::MatrixXd EV = es.eigenvectors();
    std::cout << EV.col(l) << std::endl;
    std::cout << "sanity---------------------" << std::endl;
    std::cout << Q * EV.col(l) - EV.col(l) * es.eigenvalues().row(l) << std::endl;
    return 0;
}
