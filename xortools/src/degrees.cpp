#include <iostream>
#include "xorsat.hpp"
#include "bitmatrix.hpp"
#include <sstream>
#include <vector>
#include <fstream>


int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: degrees instance.tsv" << std::endl;
        return 0;
    }
    instance I;
    std::string filename = argv[1];
    if(!load_instance(filename, I)) return 0;
    std::vector<int> var_degrees(I.num_cons+1);
    std::vector<int> con_degrees(I.num_vars+1);
    int maxvardegree = 0;
    int maxcondegree = 0;
    int degree;

    for(const unweighted_instance &UI : I.U) {
        for(const bitvector &con : UI.c) {
            degree = con.count();
            con_degrees[degree] += 1;
            if(degree > maxcondegree) maxcondegree = degree;
        }
        for(const bitvector &var : UI.v) {
            degree = var.count();
            var_degrees[degree] += 1;
            if(degree > maxvardegree) maxvardegree = degree;
        }
    }

    std::stringstream ss;
    std::vector<std::string> pages;

    ss << "\\begin{tabular}[t]{|c|c|}\n";
    ss << "\t\\hline\n";
    ss << "\t\\multicolumn{2}{|c|}{\\footnotesize variables} \\\\ \\hline\n";
    ss << "\t\\footnotesize degree & \\footnotesize count \\\\ \\hline\n";
    size_t num_printed = 0;
    for(degree = 0; degree <= maxvardegree; degree++) {
        if(var_degrees[degree] == 0) { continue; }
        ss << "\t\\footnotesize " << degree << "\t&\t\\footnotesize " << var_degrees[degree] << "\t\\\\ \\hline\n";
        if (++num_printed >= 40) {
            ss << "\\end{tabular}\n";
            pages.push_back(ss.str());
            ss.str("");
            ss.clear();
            num_printed = 0;
            ss << "\\begin{tabular}[t]{|c|c|}\n";
            ss << "\t\\hline\n";
            ss << "\t\\multicolumn{2}{|c|}{\\footnotesize variables} \\\\ \\hline\n";
            ss << "\t\\footnotesize degree & \\footnotesize count \\\\ \\hline\n";
        }
    }
    ss << "\\end{tabular}\n";
    pages.push_back(ss.str());

    ss.str("");
    ss.clear();
    ss << "\\begin{tabular}[t]{|c|c|}\n"; 
    ss << "\t\\hline\n";
    ss << "\t\\multicolumn{2}{|c|}{\\footnotesize clauses} \\\\ \\hline\n";
    ss << "\t\\footnotesize degree & \\footnotesize count \\\\ \\hline\n";

    num_printed = 0;
    for(degree = 0; degree <= maxcondegree; degree++) {
        if(con_degrees[degree] == 0) { continue; }
        ss << "\t\\footnotesize " << degree << "\t&\t\\footnotesize " << con_degrees[degree] << "\t\\\\ \\hline\n";
        if (++num_printed >= 40) {
            ss << "\\end{tabular}\n";
            pages.push_back(ss.str());
            ss.str("");
            ss.clear();
            num_printed = 0;
            ss << "\\begin{tabular}[t]{|c|c|}\n";
            ss << "\t\\hline\n";
            ss << "\t\\multicolumn{2}{|c|}{\\footnotesize clauses} \\\\ \\hline\n";
            ss << "\t\\footnotesize degree & \\footnotesize count \\\\ \\hline\n";
        }
    }
    ss << "\\end{tabular}\n";
    pages.push_back(ss.str());

    size_t cols_per_page = 5;
    // Print pages with correct wrapping in math mode
    for (size_t i = 0; i < pages.size(); ++i) {
        if (i % cols_per_page == 0) {
            if (i > 0) {
                std::cout << "\\end{array}\n\\]\n";  // Close previous array
            }
            std::cout << "\\[\n\\begin{array}{";
            size_t columns_in_page = (pages.size() - i >= cols_per_page) ? cols_per_page : pages.size() - i;
            for (size_t j = 0; j < columns_in_page; ++j) {
                std::cout << "l";
            }
            std::cout << "}\n";
        }
        if (i % cols_per_page != 0) {
            std::cout << "&\n";
        }
        std::cout << pages[i];
    }
    std::cout << "\\end{array}\n\\]\n";  // Close the final array

    std::cout << "Number of variables: " << I.num_vars << std::endl;
    std::cout << "Number of constraints: " << I.num_cons << std::endl;
    std::cout << "Variable degree counts:" << std::endl;
    for(degree = 0; degree <= maxvardegree; degree++) if(var_degrees[degree] > 0) std::cout << degree << '\t' << var_degrees[degree] << std::endl;
    std::cout << "Constraint degree counts:" << std::endl;
    for(degree = 0; degree <= maxcondegree; degree++) if(con_degrees[degree] > 0) std::cout << degree << '\t' << con_degrees[degree] << std::endl;
    size_t total = 0;
    int firstmoment = 0;
    for(degree = 0; degree <= maxvardegree; degree++) {
        if(var_degrees[degree] > 0) {
            total += var_degrees[degree];
            firstmoment += degree * var_degrees[degree];
        }
    }
    if(total != I.num_vars) std::cout << "Error: total var count not consistent." << std::endl;
    std::cout << "Average variable degree: " << (double)firstmoment/(double)total << std::endl;
    total = 0;
    firstmoment = 0;
    for(degree = 0; degree <= maxcondegree; degree++) {
        if(con_degrees[degree] > 0) {
            total += con_degrees[degree];
            firstmoment += degree * con_degrees[degree];
        }
    }
    if(total != I.num_cons) std::cout << "Error: total constraint count not consistent." << std::endl;
    std::cout << "Average constraint degree: " << (double)firstmoment/(double)total << std::endl;

{
    std::ofstream out("cons_degrees.csv");
    out<<"degree,num\n";
    for (size_t degree=0; degree<=maxcondegree; ++degree) {
        out << degree <<"," <<con_degrees[degree]<<"\n";
    }
}
{
    std::ofstream out("var_degrees.csv");
    out<<"degree,num\n";
    for (size_t degree=0; degree<=maxvardegree; ++degree) {
        out << degree <<"," <<var_degrees[degree]<<"\n";
    }
}

    return 0;
}
