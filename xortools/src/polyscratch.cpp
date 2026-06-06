//compile with: g++ -Wall -O3 --std=c++11 youngpoly.cpp -o youngpoly -lntl -lgmp -pthread
#include <Eigen/Dense>
#include <NTL/RR.h>
#include <iostream>
#include <vector>
#include "polytools.hpp"

Eigen::MatrixXd shiftcoeffs(const Eigen::MatrixXd coeffs, int m) {
    size_t l = coeffs.size()-1;
    NTL::RR u,r;
    NTL::RR v[l+1];
    NTL::ZZ num, den, mzz;
    NTL::conv(mzz, m);
    Eigen::MatrixXd vvec(1,l+1);
    for(size_t k = 0; k <= l; k++) {
        v[k] = 0.0;
        for(size_t j = k; j <= l; j++) {
            NTL::conv(u,coeffs(j));
            num = binomial(j,k) * NTL::power(mzz,j-k);
            NTL::MakeRR(r, num, -j); //r = num * 2^(-j)
            v[k] += u * r;
        }
    }
    NTL::RR shiftnorm, vn;
    double vd;
    shiftnorm = 0.0;
    for(size_t k = 0; k <= l; k++) shiftnorm += v[k] * v[k];
    shiftnorm = NTL::sqrt(shiftnorm);
    for(size_t k = 0; k <= l; k++) {
        vn = v[k] / shiftnorm;
        conv(vd, vn);
        vvec(k) = vd;
    }
    return vvec;
}

int main(int argc, char *argv[]) {
    if(argc != 5) {
        std::cout << "Usage: polyscratch m l d delta" << std::endl;
        return 0;
    }
    int m = std::stoi(argv[1]);
    int l = std::stoi(argv[2]);
    int d = std::stoi(argv[3]);
    double delta = std::stof(argv[4]);
    int listsize = 1;
    int modulus = 2;
    //std::cout << "Generating matrices" << std::endl;
    //std::cout << "A*A^T--------------------------" << std::endl;
    Eigen::MatrixXd A = generate_A(m, l, listsize, modulus);
    //std::cout << A*A.transpose() << std::endl;
    Eigen::MatrixXd D(m+1,m+1);
    for(int m_index = 0; m_index <= m; m_index++) D(m_index,m_index) = (double)m_index;
    Eigen::MatrixXd Q = A * D * A.transpose();
    //std::cout << "lambda-----------------------------" << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Q);
    std::cout << m << '\t' << l << '\t' << es.eigenvalues().row(l) << std::endl;
    std::cout << "eigenvector, Young basis:----------------------" << std::endl;
    Eigen::MatrixXd EV = es.eigenvectors();
    std::cout << EV.col(l) << std::endl;
    Eigen::MatrixXd C = convertor_matrix(m, l, listsize, modulus);
    Eigen::MatrixXd v = EV.col(l);
    Eigen::MatrixXd coeffs = v.transpose() * C;
    std::cout << "eigenvector, monomial basis:-------------------" << std::endl;
    std::cout << coeffs << std::endl;
    std::cout << "mean, according to monomial basis:-------------" << std::endl;
    std::cout << eval_mean_balanced_RR(m, coeffs) << std::endl;
    Eigen::MatrixXd scoeffs = shiftcoeffs(coeffs, m);
    std::cout << "eigenvector, shifted basis:-------------------" << std::endl;
    std::cout << scoeffs << std::endl;
    std::cout << "mean, according to shifted basis:-------------" << std::endl;
    std::cout << eval_mean_shifted(m, scoeffs) << std::endl;
    std::cout << "mean, evaluated by moments:-------------------" << std::endl;
    std::cout << eval_s_by_moments(m, scoeffs) << std::endl;
    std::cout << "lower bound:-------------------" << std::endl;
    std::cout << eval_sbound(m, scoeffs, delta, d) << std::endl;
    //std::cout << "matrix*eigenvector---------------------" << std::endl;
    //std::cout << Q * EV.col(l) << std::endl;
    //std::cout << "lambda*eigenvector---------------------" << std::endl;
    //std::cout << EV.col(l) * es.eigenvalues().row(l) << std::endl;
    return 0;
}

//This is for testing Adam Zalcman's formula. His formula is indeed confirmed.

/*std::vector<double> viathor_coeffs_rhalf(int m, int l) {
    NTL::ZZ normsq;
    normsq = 0;
    int k;
    for(k = 0; k <= l; k++) normsq += binomial(m, k);
    std::vector<double> returnval(l+1);
    for(k = 0; k <= l; k++) {
        double exponent = 0.5 * (log(binomial(m,k)) - log(normsq));
        returnval[k] = exp(exponent);
    }
    return returnval;
}

int main(int argc, char *argv[]) {
    if(argc != 3) {
        std::cout << "Usage: polyscratch m l" << std::endl;
        return 0;
    }
    int m = std::stoi(argv[1]);
    int l = std::stoi(argv[2]);
    int t = m;
    int listsize = 1;
    int modulus = 2;
    std::cout << "Generating matrices" << std::endl;
    std::cout << "A*A^T--------------------------" << std::endl;
    Eigen::MatrixXd A = generate_A(m, l, listsize, modulus);
    std::cout << A*A.transpose() << std::endl;
    std::cout << "Agt--------------------------------" << std::endl;
    Eigen::MatrixXd Agt = A.rightCols(m-t+1);
    std::cout << Agt << std::endl;
    std::cout << "Agt normalized-----------------------" << std::endl;
    Eigen::VectorXd vn = A.col(m);
    vn.normalize();
    std::cout << vn << std::endl;
    std::cout << "t,lambda-----------------------------" << std::endl;
    Eigen::MatrixXd AAgt = Agt*Agt.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AAgt);
    std::cout << t << "\t" << es.eigenvalues().row(l) << std::endl;
    std::cout << "eigenvector---------------------------" << std::endl;
    Eigen::MatrixXd EV = es.eigenvectors();
    std::cout << EV.col(l) << std::endl;
    std::cout << "matrix*eigenvector---------------------" << std::endl;
    std::cout << AAgt * EV.col(l) << std::endl;
    std::cout << "lambda*eigenvector---------------------" << std::endl;
    std::cout << EV.col(l) * es.eigenvalues().row(l) << std::endl;
    std::cout << "adams_vector---------------------------" << std::endl;
    std::vector<double> vc = viathor_coeffs_rhalf(m, l);
    for(double d : vc) std::cout << d << "\n";
    return 0;
}*/
