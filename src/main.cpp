#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "../include/wigner.hpp"
#include "../include/bose_sys.hpp"

void print_csv_header(int nsites) {
    std::cout << "t,";
    for (int i = 0; i < nsites; i++) {
        std::cout << "p" << i;
        if (i != nsites - 1) std::cout << ",";
    }
    std::cout << "\n";
}

void print_csv_row(double t, Eigen::VectorXd population) {
    std::cout << t << ",";
    for (int i = 0; i < population.size(); i++) {
        std::cout << population(i);
        if (i != population.size() - 1) std::cout << ",";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Too few arguments.\n";
        exit(-1);
    }

    std::ifstream input_file(argv[1]);

    int num_sites, num_samples, num_steps;
    double init_population, dt, t;
    Eigen::MatrixXcd new_field, old_field;
    Eigen::VectorXd population;

    input_file >> num_sites;
    input_file >> num_samples;
    input_file >> init_population;

    bose_system_t bose;
    bose.n_sites = num_sites;
    input_file >> bose.g;
    input_file >> bose.t;
    
    population = Eigen::VectorXd::Zero(num_sites);
    population(num_sites / 2) = init_population;

    old_field = set_initial_condition(num_sites, num_samples, population);
    new_field = Eigen::MatrixXcd::Zero(num_sites, num_samples);

    population = avg_pop(old_field);

    print_csv_header(num_sites);
    num_steps = 200;
    input_file >> dt;
    t = 0.;
    for (int i = 0; i < num_steps; i++) {
        print_csv_row(t, population);
        step_forward(old_field, new_field, bose, dt);
        old_field = new_field;
        population = avg_pop(old_field);
        t += dt;
    }
    
    return 0;
}