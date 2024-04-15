#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <mpi.h>
#include "hdf5.h"
#include "../include/get_walltime.h"
#include "../include/wigner.hpp"
#include "../include/bose_sys.hpp"

#define BUFFSIZE 256

void print_csv_header(int nsites) {
    std::cout << "t,";
    for (int i = 0; i < nsites; i++) {
        std::cout << "p" << i;
        std::cout << ",";
    }
    std::cout << "walltime" << "\n";
}

void print_csv_row(double t, Eigen::VectorXd population, double walltime) {
    std::cout << t << ",";
    for (int i = 0; i < population.size(); i++) {
        std::cout << population(i);
        std::cout << ",";
    }
    std::cout << walltime;
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        std::cerr << "Too few arguments.\n";
        exit(-1);
    }

    hid_t file, dset;
    char dset_name[BUFFSIZE];
    int num_sites, num_samples, num_steps;
    double init_population, dt, t, end_time, start_time;
    Eigen::MatrixXcd new_field, old_field;
    Eigen::VectorXd population, times;
    Eigen::MatrixXd all_populations;

	num_samples = 0;
    std::ifstream input_file(argv[1]);
    if (rank == 0) {
        input_file >> num_sites;
        input_file >> num_samples;
        input_file >> init_population;
    }
    MPI_Bcast(&num_sites, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&init_population, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //std::cerr << "rank " << rank << " has sitee " << num_sites << " and samples " << num_samples << std::endl;

    bose_system_t bose;
    bose.n_sites = num_sites;
    if (rank == 0) {
        input_file >> bose.g;
        input_file >> bose.t;
    }
    MPI_Bcast(&bose.g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bose.t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //print_csv_header(num_sites);
    //num_steps = 200;
    if (rank == 0) {
        input_file >> num_steps;
        input_file >> dt;
    }
    MPI_Bcast(&num_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) print_csv_header(num_sites);

    /* split the samples between all the rank. 
     * In the case where the number is not evenly divisible,
     * Divide the remainder amongst some of the nodes to compensate.
     */
	//std::cerr << "num_samples = " << num_samples << std::endl;
    int num_samples_this_rank = num_samples / world_size;
    int remainder = num_samples - world_size * (num_samples / world_size);
	//if (rank == 0) std::cerr << remainder << std::endl;
    if (rank < remainder) num_samples_this_rank++;
	//std::cerr << rank << " " << num_samples_this_rank << std::endl;
    int num_samples_all_ranks;
    MPI_Allreduce(&num_samples_this_rank, &num_samples_all_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (num_samples_all_ranks != num_samples) {
        if (rank == 0) std::cerr << "Load balanced total " << num_samples_all_ranks 
            << " does not match total " << num_samples << ".\n";
        exit(-1);
    }

    population = Eigen::VectorXd::Zero(num_sites);
    population(num_sites / 2) = init_population;

    old_field = set_initial_condition(num_sites, num_samples_this_rank, population);
    new_field = Eigen::MatrixXcd::Zero(num_sites, num_samples_this_rank);

    population = avg_pop(old_field);


    t = 0.;
    for (int i = 0; i < num_steps; i++) {
        get_walltime(&start_time);
        step_forward(old_field, new_field, bose, dt);
        old_field = new_field;
        population = avg_pop(old_field, rank, world_size, true);
        t += dt;
        get_walltime(&end_time);
        if (rank == 0) print_csv_row(t, population, end_time - start_time);
    }
    
    MPI_Finalize();
}
