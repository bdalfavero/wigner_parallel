#include <iostream>
#include <random>
#include <cmath>
#include <complex>
#include <mpi.h>
#include <Eigen/Dense>
#include <omp.h>
#include "../include/wigner.hpp"

Eigen::MatrixXcd set_initial_condition(int num_sites, int num_samples, Eigen::VectorXd init_pop) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    Eigen::MatrixXcd psi;
    double yr, yi;

    psi = Eigen::MatrixXcd::Zero(num_sites, num_samples);
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_samples; j++) {
            yr = distribution(generator);
            yi = distribution(generator);
            psi(i, j) = 0.5 * std::complex<double>(yr, yi) + std::complex<double>(sqrt(init_pop(i)), 0.0);
        }
    }

    return psi;
}

void step_forward(Eigen::MatrixXcd old_field, Eigen::MatrixXcd &new_field, bose_system_t bose, double dt, bool use_simd){
    int nsites, nsamples;
    std::complex<double> rhs;

    if (new_field.rows() != bose.n_sites) {
        std::cerr << "Number of sites does not match between bose system and field in time step.\n";
        exit(-1);
    }

    if ((old_field.rows() != new_field.rows()) || (old_field.cols() != new_field.cols())) {
        std::cerr << "Field dimensions do not match.\n";
        exit(-1);
    }

    nsites = old_field.rows();
    nsamples = old_field.cols();

	if (!use_simd) {
		#pragma omp parallel 
		{
			#pragma omp for
			for (int j = 0; j < nsamples; j++) {
				for (int i = 0; i < nsites; i++) {
					//rhs = std::complex<double>(0., 0.);
					rhs = std::complex<double>(-bose.g, 0.) * (old_field(i,j) + std::complex<double>(pow(abs(old_field(i, j)), 2)), 0.) * old_field(i, j);
					if (i != 0) rhs += std::complex<double>(bose.t, 0.) * old_field(i - 1, j);
					if (i != nsites - 1) rhs += std::complex<double>(bose.t, 0.) * old_field(i + 1, j);
					new_field(i, j) = old_field(i, j) + std::complex<double>(0., -dt) * rhs; 
				}
			}
		}
	} else {
		#pragma omp parallel 
		{
			#pragma omp for
			for (int j = 0; j < nsamples; j++) {
				#pragma omp simd
				for (int i = 0; i < nsites; i++) {
					rhs = std::complex<double>(-bose.g, 0.) * (old_field(i,j) + std::complex<double>(pow(abs(old_field(i, j)), 2)), 0.) * old_field(i, j);
					new_field(i, j) = old_field(i, j) + std::complex<double>(0., -dt) * rhs; 
				}
				for (int i = 0; i < nsites; i++) {
					rhs = std::complex<double>(0., 0.);
					if (i != 0) rhs += std::complex<double>(bose.t, 0.) * old_field(i - 1, j);
					if (i != nsites - 1) rhs += std::complex<double>(bose.t, 0.) * old_field(i + 1, j);
					new_field(i, j) = old_field(i, j) + std::complex<double>(0., -dt) * rhs; 
				}
			}
		}

	}
    return;
}

Eigen::VectorXd avg_pop(Eigen::MatrixXcd field, int rank, int world_size, bool parallel) {
    if (!parallel) {
        Eigen::VectorXd population;
        int nsites, nsamples;

        nsites = field.rows();
        nsamples = field.cols();
        population = Eigen::VectorXd::Zero(nsites);

		#pragma omp parallel
		{
			#pragma omp for
			for (int i = 0; i < nsites; i++) {
				population(i) = field.row(i).squaredNorm() / (double)nsamples;
			}
		}
        return population;
    } else {
        //std::cerr << "In averaging function." << std::endl;
        //std::cerr << rank << " " << field.rows() << " " << field.cols() << std::endl;
        Eigen::VectorXd this_population, total_population;
        int nsites, this_nsamples, total_nsamples;

        nsites = field.rows();
        this_nsamples = field.cols();
        total_nsamples = 0;
        this_population = Eigen::VectorXd::Zero(nsites);
        total_population = Eigen::VectorXd::Zero(nsites);

        /* Add together all samples FOR THIS NODE */
        for (int i = 0; i < nsites; i++) {
            this_population(i) = field.row(i).squaredNorm();
        }

        /* Get total number of samples for all nodes. */
        MPI_Allreduce(&this_nsamples, &total_nsamples, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        /* Add together all populations from each nodes. */
        MPI_Allreduce(this_population.data(), total_population.data(), nsites, 
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        total_population = total_population / (double)total_nsamples;

        return total_population;
    }
}
