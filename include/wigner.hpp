#ifndef WIGNER
#define WIGNER

#include "bose_sys.hpp"

Eigen::MatrixXcd set_initial_condition(int num_sites, int num_samples, Eigen::VectorXd init_pop);

void step_forward(Eigen::MatrixXcd old_field, Eigen::MatrixXcd &new_field, bose_system_t bose, double dt);

Eigen::VectorXd avg_pop(Eigen::MatrixXcd field, int rank = 0, int world_size = 0, bool parallel = 0);

#endif