#ifndef POLYTOOLS_HPP
#define POLYTOOLS_HPP

#include <NTL/ZZ.h>
#include <NTL/RR.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>

std::string vecstring(const std::vector<NTL::ZZ> &v) {
    std::stringstream buf;
    buf <<  v.at(0);
    for(size_t i = 1; i < v.size(); i++) buf << "," << v.at(i);
    return buf.str();
}

NTL::ZZ binomial(int n, int k) {
    NTL::ZZ returnval;
    if(n == 0) {
        returnval = 0;
        if(k == 0) returnval = 1;
        return returnval;
    }
    if(n < 0) {
        returnval = 0;
        std::cout << "Error: binomial has not been implemented for n = " << n << " k = " << k << std::endl;
        return returnval;
    }
    if(k > n) {
        returnval = 0;
        return returnval;
    }
    if(k == 0 || k == n) {
        returnval = 1;
        return returnval;
    }
    NTL::ZZ denominator;
    denominator = 1;
    returnval = 1;
    for(int i = k+1; i <= n; i++) returnval *= i;
    for(int i = 1; i <= n-k; i++) denominator *= i;
    returnval /= denominator;
    return returnval;
}

/*
  These are the coefficients for a set of polynomials that are orthogonal according
  to the binomial measure. Modulo normalization, they are the polynomials from:
  Tzay Y. Young. Binomial-Weighted Orthogonal Polynomials
  Journal of the Association for Computing Machinery
  Vol 14, No. 1, pg 120-127. 1967

  See notes for definition of r, rho, mu.
*/
std::vector<NTL::ZZ> youngpoly(int n, int r, int rho, int mu) {
    if(n < 0 || n > 1E6 || rho <= 0 || rho >= mu) {
        std::cout << "parameters out of range in youngpoly" << std::endl;
        std::vector<NTL::ZZ> empty;
        return empty;
    }
    std::vector<NTL::ZZ> a(n+1);
    std::vector<NTL::ZZ> b(n+1);
    std::vector<NTL::ZZ> c(n+1);
    std::vector<NTL::ZZ> *current;
    std::vector<NTL::ZZ> *back1;
    std::vector<NTL::ZZ> *back2;
    std::vector<NTL::ZZ> *tmp;
    NTL::ZZ n_z, rho_z, theta_z, mu_z;
    rho_z = rho;
    theta_z = mu - rho;
    mu_z = mu;
    n_z = n;
    a.at(0) = 1;
    if(r == 0) return a;
    b.at(0) = -n_z*rho_z;
    b.at(1) = mu_z;
    if(r == 1) return b;
    back2 = &c;
    current = &b;
    back1 = &a;
    for(int rp = 2; rp <= r; rp++) {
        //cyclically shift the pointers---------
        tmp = current;
        current = back2;
        back2 = back1;
        back1 = tmp;
        //--------------------------------------
        for(int i = 0; i <= rp; i++) current->at(i) = ((rp-1)*(rho_z-theta_z)-n*rho_z) * back1->at(i) - rho_z*theta_z*(rp-1)*(n-rp+2)*back2->at(i);
        for(int i = 1; i <= rp; i++) current->at(i) += mu_z*back1->at(i-1);
    }
    return *current;
}

std::vector <std::vector<NTL::ZZ>> all_youngpoly(int m, int l, int listsize, int modulus) {
    if(m < 0 || m > 1E6 || listsize <= 0 || listsize >= modulus) {
        std::cout << "parameters out of range in youngpoly" << std::endl;
        std::vector <std::vector<NTL::ZZ>> empty;
        return empty;
    }
    std::vector< std::vector<NTL::ZZ> > all(l+1, std::vector<NTL::ZZ>(l+1));
    NTL::ZZ n_z, rho_z, theta_z, mu_z;
    rho_z = listsize;
    theta_z = modulus - listsize;
    mu_z = modulus;
    n_z = m;
    all[0][0] = 1;
    if(l == 0) return all;
    all[1][0] = -n_z*rho_z;
    all[1][1] = mu_z;
    if(l == 1) return all;
    for(int rp = 2; rp <= l; rp++) {
        for(int i = 0; i <= rp; i++) all[rp][i] = ((rp-1)*(rho_z-theta_z)-n_z*rho_z) * all[rp-1][i] - rho_z*theta_z*(rp-1)*(n_z-rp+2)*all[rp-2][i];
        for(int i = 1; i <= rp; i++) all[rp][i] += mu_z*all[rp-1][i-1];
    }
    return all;
}

NTL::ZZ eval_poly(NTL::ZZ x, const std::vector<NTL::ZZ> &coeffs) {
    NTL::ZZ monomial;
    NTL::ZZ value;
    monomial = 1.0;
    value = 0.0;
    for(size_t power = 0; power < coeffs.size(); power++) {
        value += coeffs.at(power)*monomial;
        monomial *= x;
    }
    return value;
}

NTL::ZZ normsq(const std::vector<NTL::ZZ> &coeffs, std::vector<NTL::ZZ> imeasure) {
    NTL::ZZ returnval, x, val;
    returnval = 0;
    for(size_t i = 0; i < imeasure.size(); i++) {
        x = i;
        val = eval_poly(x, coeffs);
        returnval += val*val*imeasure.at(i);
    }
    return returnval;
}

//right multiply by M to go from coefficients on young polynomials to coefficients on monomials
Eigen::MatrixXd convertor_matrix(int m, int l, int listsize, int modulus) {
    if(m < 0 || m > 1E6 || listsize <= 0 || listsize >= modulus) {
        std::cout << "parameters out of range in convertor_matrix" << std::endl;
        Eigen::MatrixXd empty;
        return empty;
    }
    NTL::ZZ absentry;
    double sign, exponent;
    std::vector <std::vector<NTL::ZZ>> all = all_youngpoly(m,l,listsize,modulus);
    std::vector<NTL::ZZ> imeasure(m+1);
    NTL::ZZ rho_z, theta_z;
    rho_z = listsize;
    theta_z = modulus - listsize;
    for(int j = 0; j <=m; j++) imeasure[j] = binomial(m,j) * power(rho_z, j) * power(theta_z, m-j);
    std::vector<NTL::ZZ> normsqs(l+1);
    for(int rp = 0; rp <=l; rp++) normsqs[rp] = normsq(all[rp], imeasure);
    Eigen::MatrixXd M(l+1,l+1);
    for(int rp = 0; rp <= l; rp++) {
        for(int i = 0; i <= l; i++) {
            if(all[rp][i] < 0) {
                sign = -1;
                absentry = -all[rp][i];
            }
            if(all[rp][i] > 0) {
                sign = 1;
                absentry = all[rp][i];
            }
            if(all[rp][i] == 0) M(rp,i) = 0.0;
            else {
                exponent = log(absentry) - 0.5 * log(normsqs[rp]);
                M(rp,i) = sign*exp(exponent);
            }
        }
    }
    return M;
}

//right multiply by M to go from coefficients on young polynomials to coefficients on monomials
//this one normalizes the matrix so that its largest entry is one
//this did not help
/*Eigen::MatrixXd convertor_matrix_n(int m, int l, int listsize, int modulus) {
    if(m < 0 || m > 1E6 || listsize <= 0 || listsize >= modulus) {
        std::cout << "parameters out of range in convertor_matrix" << std::endl;
        Eigen::MatrixXd empty;
        return empty;
    }
    NTL::ZZ absentry;
    double sign, exponent;
    std::vector <std::vector<NTL::ZZ>> all = all_youngpoly(m,l,listsize,modulus);
    std::vector<NTL::ZZ> imeasure(m+1);
    NTL::ZZ rho_z, theta_z;
    rho_z = listsize;
    theta_z = modulus - listsize;
    for(int j = 0; j <=m; j++) imeasure[j] = binomial(m,j) * power(rho_z, j) * power(theta_z, m-j);
    std::vector<NTL::ZZ> normsqs(l+1);
    for(int rp = 0; rp <=l; rp++) normsqs[rp] = normsq(all[rp], imeasure);
    Eigen::MatrixXd M(l+1,l+1);
    double maxexp = -1E6;
    for(int rp = 0; rp <= l; rp++) {
        for(int i = 0; i <= l; i++) {
            if(all[rp][i] < 0) absentry = -all[rp][i];
            else absentry = all[rp][i];
            if(absentry > 0) {
                exponent = log(absentry) - 0.5 * log(normsqs[rp]);
                if(exponent > maxexp) maxexp = exponent;
            }
        }
    }
    for(int rp = 0; rp <= l; rp++) {
        for(int i = 0; i <= l; i++) {
            if(all[rp][i] < 0) {
                sign = -1;
                absentry = -all[rp][i];
            }
            if(all[rp][i] > 0) {
                sign = 1;
                absentry = all[rp][i];
            }
            if(all[rp][i] == 0) M(rp,i) = 0.0;
            else {
                exponent = log(absentry) - 0.5 * log(normsqs[rp]);
                M(rp,i) = sign*exp(exponent - maxexp);
            }
        }
    }
    return M;
}*/

//be careful of overflow!
double eval_poly(double x, const Eigen::MatrixXd &coeffs) {
    double monomial;
    double value;
    monomial = 1.0;
    value = 0.0;
    for(size_t power = 0; power < coeffs.size(); power++) {
        value += coeffs(0,power)*monomial;
        monomial *= x;
    }
    return value;
}

NTL::RR eval_poly_RR(NTL::RR x, const Eigen::MatrixXd &coeffs) {
    int l = coeffs.size() - 1;
    std::vector<NTL::RR>rrcoeffs(l+1);
    for(size_t k = 0; k <= l; k++) NTL::conv(rrcoeffs[k], coeffs(k));
    NTL::RR monomial;
    NTL::RR value;
    monomial = 1.0;
    value = 0.0;
    for(size_t power = 0; power < coeffs.size(); power++) {
        value += coeffs(0,power)*monomial;
        monomial *= x;
    }
    return value;
}

//be careful of overflow!
double eval_mean_balanced(int m, const Eigen::MatrixXd &coeffs) {
    std::vector<double> measure(m+1);
    double exponent;
    for(int t = 0; t <= m; t++) {
        exponent = log(binomial(m,t)) - m*log(2.0);
        measure[t] = exp(exponent);
    }
    double numerator = 0.0;
    double denominator = 0.0;
    double P;
    for(int t = 0; t <= m; t++) {
        P = eval_poly((double)t, coeffs);
        numerator += measure[t] * P * P * (double)t;
        denominator += measure[t] * P * P;
    }
    return numerator/denominator;
}

double eval_mean_balanced_RR(int m, const Eigen::MatrixXd &coeffs) {
    std::vector<NTL::RR> measure(m+1);
    for(int t = 0; t <= m; t++) NTL::MakeRR(measure[t], binomial(m,t),-m);
    NTL::RR numerator, denominator, P, x;
    numerator = 0.0;
    denominator = 0.0;
    for(int t = 0; t <= m; t++) {
        NTL::conv(x, t);
        P = eval_poly_RR(x, coeffs);
        numerator += measure[t] * P * P * x;
        denominator += measure[t] * P * P;
    }
    NTL::RR result;
    result = numerator/denominator;
    double returnval;
    NTL::conv(returnval, result);
    return returnval;
}

double eval_mean_shifted(int m, const Eigen::MatrixXd &scoeffs) {
    std::vector<double> measure(m+1);
    double exponent;
    for(int t = 0; t <= m; t++) {
        exponent = log(binomial(m,t)) - m*log(2.0);
        measure[t] = exp(exponent);
    }
    double numerator = 0.0;
    double denominator = 0.0;
    double P;
    for(int t = 0; t <= m; t++) {
        P = eval_poly((double)(2*t-m), scoeffs);
        numerator += measure[t] * P * P * t;
        denominator += measure[t] * P * P;
    }
    return numerator/denominator;
}

//given coefficients of univariate polynomial P in ascending order by degree
//returns coefficients of P^2 from lowest to highest degree
Eigen::MatrixXd psquared_coeffs(const Eigen::MatrixXd &coeffs) {
    int l = coeffs.size()-1;
    Eigen::MatrixXd sqcoeffs(1,2*l+1);
    for(int k = 0; k <= 2*l; k++) sqcoeffs(k) = 0.0;
    for(int i = 0; i <= l; i++) {
        for(int j = 0; j <= l; j++) {
            sqcoeffs(i+j) += coeffs(i) * coeffs(j);
        }
    }
    return sqcoeffs;
}

//this will return expected number satisfied minus expected number unsatisfied
//input must be shifted coeffs
double eval_f_by_moments(int m, const Eigen::MatrixXd &scoeffs) {
    size_t l = scoeffs.size()-1;
    std::vector<NTL::RR> measure(m+1);
    for(int t = 0; t <= m; t++) {
        NTL::MakeRR(measure[t], binomial(m,t), -m); //r = binomial(m,t) * 2^(-m)
    }
    std::vector<NTL::RR> moment(2*l+2); //we'll use the centered moments
    moment[0] = 1.0;
    for(int p = 1; p <= 2*l+1; p++) {
        moment[p] = 0.0;
        for(int t = 0; t <= m; t++) {
            NTL::RR base;
            NTL::conv(base, 2*t-m); //base = 2*t-m
            moment[p] += measure[t] * NTL::power(base,p);
        }
    }
    Eigen::MatrixXd psqcoeffs = psquared_coeffs(scoeffs);
    NTL::RR numerator;
    NTL::RR denominator;
    numerator = 0.0;
    denominator = 0.0;
    for(int p = 0; p <= 2*l; p++) {
        numerator += psqcoeffs(p) * moment[p+1];
        denominator += psqcoeffs(p) * moment[p];
    }
    NTL::RR ratio;
    ratio =  numerator / denominator;
    double returnval;
    NTL::conv(returnval, ratio);
    return returnval;
}

//this will return expected number satisfied
//input must be shifted coeffs
double eval_s_by_moments(int m, const Eigen::MatrixXd &scoeffs) {
    double f = eval_f_by_moments(m, scoeffs);
    return 0.5*(f+(double)m);
}

//this will return expected number satisfied minus expected number unsatisfied
//input must be shifted coeffs, if the moments can be off by epsilon
double eval_fbound(int m, const Eigen::MatrixXd &scoeffs, double delta, int distance) {
    size_t l = scoeffs.size()-1;
    std::vector<NTL::RR> measure(m+1);
    for(int t = 0; t <= m; t++) {
        NTL::MakeRR(measure[t], binomial(m,t), -m); //r = binomial(m,t) * 2^(-m)
    }
    std::vector<NTL::RR> moment(2*l+2); //we'll use the centered moments
    moment[0] = 1.0;
    for(int p = 1; p <= 2*l+1; p++) {
        moment[p] = 0.0;
        for(int t = 0; t <= m; t++) {
            NTL::RR base;
            NTL::conv(base, 2*t-m); //base = 2*t-m
            moment[p] += measure[t] * NTL::power(base,p);
        }
    }
    Eigen::MatrixXd psqcoeffs = psquared_coeffs(scoeffs);
    NTL::RR numerator;
    NTL::RR denominator;
    NTL::RR product;
    numerator = 0.0;
    denominator = 0.0;
    for(int p = 0; p <= 2*l; p++) {
        if(p < distance) {
            numerator += psqcoeffs(p) * moment[p+1];
            denominator += psqcoeffs(p) * moment[p];
        }
        else {
            product = psqcoeffs(p) * moment[p+1];
            if(product > 0.0) numerator += product * (1.0 - delta);
            else numerator += product * (1.0 + delta);
            product = psqcoeffs(p) * moment[p];
            if(product > 0.0) denominator += product * (1.0 + delta);
            else denominator += product * (1.0 - delta);
        }
    }
    NTL::RR ratio;
    ratio =  numerator / denominator;
    double returnval;
    NTL::conv(returnval, ratio);
    return returnval;
}

//this will return expected number satisfied
//input must be shifted coeffs
double eval_sbound(int m, const Eigen::MatrixXd &scoeffs, double delta, int distance) {
    double f = eval_fbound(m, scoeffs, delta, distance);
    return 0.5*(f+(double)m);
}

Eigen::MatrixXd generate_A(int m, int l, int listsize, int modulus) {
    std::vector <std::vector<NTL::ZZ>> all = all_youngpoly(m, l, listsize, modulus);
    std::vector<NTL::ZZ> imeasure(m+1);
    NTL::ZZ rho_z, theta_z;
    rho_z = listsize;
    theta_z = modulus - listsize;
    for(int j = 0; j <=m; j++) imeasure[j] = binomial(m,j) * power(rho_z, j) * power(theta_z, m-j);
    std::vector<NTL::ZZ> normsqs(l+1);
    for(int i = 0; i <=l; i++) normsqs[i] = normsq(all[i], imeasure);
    Eigen::MatrixXd A(l+1,m+1);
    NTL::ZZ x, polyval;
    for(int i = 0; i <= l; i++) {
        for(int j = 0; j <= m; j++) {
            x = j;
            polyval = eval_poly(x, all[i]);
            int sign = 1;
            if(polyval < 0) {
                sign = -1;
                polyval *= sign;
            }
            if(polyval == 0) A(i,j) = 0.0;
            else {
                double exponent = log(polyval) + 0.5 * log(imeasure[j]) - 0.5 * log(normsqs[i]);
                A(i,j) = exp( exponent ) * (double)sign;
            }
        }
    }
    return A;
}

#endif