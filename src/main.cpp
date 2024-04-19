#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <mpi.h>
#include "hdf5.h"
#include "../include/get_walltime.h"
#include "../include/wigner.hpp"
#include "../include/bose_sys.hpp"

#define BUFFSIZE 256

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
    int status;
    int num_sites, num_samples, num_steps;
    double init_population, dt, t, end_time2, end_time, start_time;
	double global_start_time, global_end_time;
	int use_simd;
    Eigen::MatrixXcd new_field, old_field;
    Eigen::VectorXd population, times;
    Eigen::MatrixXd all_populations;
	Eigen::VectorXd walltimes, step_times, averaging_times;

	get_walltime(&global_start_time);

	num_samples = 0;
    //std::ifstream input_file(argv[1]);
    file = H5Fopen(argv[1], H5F_ACC_RDWR, H5P_DEFAULT);
    if (file == H5I_INVALID_HID) {
        std::cerr << "Error opening file.\n";
        return -1;
    }
    if (rank == 0) {
        //input_file >> num_sites;
        strncpy(dset_name, "num_points", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_sites);
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
        //input_file >> num_samples;
        strncpy(dset_name, "num_samples", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_samples);
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
        //input_file >> init_population;
        strncpy(dset_name, "init_population", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &init_population);
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
        strncpy(dset_name, "use_simd", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &use_simd);
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
    }
    MPI_Bcast(&num_sites, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&init_population, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_simd, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //std::cerr << "rank " << rank << " has sitee " << num_sites << " and samples " << num_samples << std::endl;

    bose_system_t bose;
    bose.n_sites = num_sites;
    if (rank == 0) {
        //input_file >> bose.g;
        strncpy(dset_name, "bose_model/g", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(bose.g));
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
        //input_file >> bose.t;
        strncpy(dset_name, "bose_model/t", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(bose.t));
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
    }
    MPI_Bcast(&bose.g, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bose.t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //print_csv_header(num_sites);
    //num_steps = 200;
    if (rank == 0) {
        //input_file >> num_steps;
        strncpy(dset_name, "time_steps/num_steps", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_steps);
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
        //input_file >> dt;
        strncpy(dset_name, "time_steps/dt", BUFFSIZE);
        dset = H5Dopen2(file, dset_name, H5P_DEFAULT);
        if (dset == H5I_INVALID_HID) {
            std::cerr << "Error opening dataset" << dset_name << ".\n";
            return -1;
        }
        status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dt);
        if (status < 0) {
            std::cerr << "Error reading dataset" << dset_name << ".\n";
            return -1;
        }
    }
    MPI_Bcast(&num_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //if (rank == 0) print_csv_header(num_sites);

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
	all_populations = Eigen::MatrixXd::Zero(num_steps, num_sites);
	step_times = Eigen::VectorXd::Zero(num_steps);
	averaging_times = Eigen::VectorXd::Zero(num_steps);
	walltimes = Eigen::VectorXd::Zero(num_steps);


    t = 0.;
    for (int i = 0; i < num_steps; i++) {
        get_walltime(&start_time);
        step_forward(old_field, new_field, bose, dt, (bool)use_simd);
        old_field = new_field;
        get_walltime(&end_time);
        population = avg_pop(old_field, rank, world_size, true);
        t += dt;
        get_walltime(&end_time2);
        //if (rank == 0) print_csv_row(t, population, end_time - start_time);
        step_times(i) = end_time - start_time;
		averaging_times(i) = end_time2 - end_time;
		walltimes(i) = end_time2 - start_time;
		all_populations.row(i) = population.transpose();
    }

	/* Write data to the file. */
	hsize_t dims[2] = {num_steps, num_sites};
	hid_t fspace = H5Screate_simple(2, dims, NULL);
	dset = H5Dcreate(file, "/population", H5T_NATIVE_DOUBLE, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dset == H5I_INVALID_HID) {
		std::cerr << "Error opening dataset /population.\n";
		return -1;
	}
	status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, all_populations.data());
	if (status < 0) {
		std::cerr << "Error writing to dataset population\n";
		return -1;
	}

	hsize_t dim = (hsize_t)num_steps;
	fspace = H5Screate_simple(1, &dim, NULL);
	dset = H5Dcreate(file, "/walltimes", H5T_NATIVE_DOUBLE, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dset == H5I_INVALID_HID) {
		std::cerr << "Error opening dataset /walltimes.\n";
		return -1;
	}
	status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, walltimes.data());
	if (status < 0) {
		std::cerr << "Error writing to dataset walltimes\n";
		return -1;
	}

	dim = (hsize_t)num_steps;
	fspace = H5Screate_simple(1, &dim, NULL);
	dset = H5Dcreate(file, "/step_times", H5T_NATIVE_DOUBLE, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dset == H5I_INVALID_HID) {
		std::cerr << "Error opening dataset /step_times.\n";
		return -1;
	}
	status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, step_times.data());
	if (status < 0) {
		std::cerr << "Error writing to dataset step_timesn";
		return -1;
	}

	dim = (hsize_t)num_steps;
	fspace = H5Screate_simple(1, &dim, NULL);
	dset = H5Dcreate(file, "/averaging_times", H5T_NATIVE_DOUBLE, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dset == H5I_INVALID_HID) {
		std::cerr << "Error opening dataset /averaging_times.\n";
		return -1;
	}
	status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, averaging_times.data());
	if (status < 0) {
		std::cerr << "Error writing to dataset averaging_times\n";
		return -1;
	}

	get_walltime(&global_end_time);
	double total_wtime = global_end_time - global_start_time;
	dim = 1;
	fspace = H5Screate_simple(1, &dim, NULL);
	dset = H5Dcreate(file, "/total_walltime", H5T_NATIVE_DOUBLE, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dset == H5I_INVALID_HID) {
		std::cerr << "Error opening dataset /total_walltime.\n";
		return -1;
	}
	status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total_wtime);
	if (status < 0) {
		std::cerr << "Error writing to dataset total_walltime\n";
		return -1;
	}

	status = H5Fclose(file);
    
    MPI_Finalize();
}
