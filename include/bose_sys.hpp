#ifndef BOSE
#define BOSE

typedef struct bose_s {
    int n_sites;
    double g; // Interaction strength
    double t; // Hopping energy
} bose_system_t;

#endif