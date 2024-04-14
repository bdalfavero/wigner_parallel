#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "hdf5.h"
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

    hid_t file, dset;

    int num_sites, num_samples, num_steps;
    double init_population, dt, t;
    Eigen::MatrixXcd new_field, old_field;
    Eigen::VectorXd population;

    if ((file = H5Fopen(argv[1], H5F_ACC_RDWR, H5P_DEFAULT)) == H5I_INVALID_HID) {
        std::cerr << "Error opening file.\n";
        exit(-1);
    }

    //input_file >> num_sites;
    if (H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_sites) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }
    //input_file >> num_samples;
    if (H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_samples) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }
    //input_file >> init_population;
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &init_population) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }

    bose_system_t bose;
    bose.n_sites = num_sites;
    //input_file >> bose.g;
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bose.g) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }
    //input_file >> bose.t;
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bose.t) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }

    population = Eigen::VectorXd::Zero(num_sites);
    population(num_sites / 2) = init_population;

    old_field = set_initial_condition(num_sites, num_samples, population);
    new_field = Eigen::MatrixXcd::Zero(num_sites, num_samples);

    population = avg_pop(old_field);

    print_csv_header(num_sites);
    //num_steps = 200;
    if (H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_steps) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }
    //input_file >> dt;
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &bose.t) < 0) {
        std::cerr << "Error opening dataset.\n";
        exit(-1);
    }
    
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