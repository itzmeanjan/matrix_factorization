#include <iostream>
#include <random>

void random_matrix(float *const matrix, const uint dim);

void show(const float *mat, const uint dim);

uint compute_work_group_size(const uint dim, uint wg_size);
