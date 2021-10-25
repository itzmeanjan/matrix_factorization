#include "utils.hpp"

void random_matrix(float *const matrix, const uint dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  for (uint i = 0; i < dim; i++) {
    for (uint j = 0; j < dim; j++) {
      matrix[i * dim + j] = dis(gen);
    }
  }
}

void show(const float *mat, const uint dim) {
  for (uint i = 0; i < dim; i++) {
    for (uint j = 0; j < dim; j++) {
      std::cout << mat[i * dim + j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

uint compute_work_group_size(const uint dim, uint wg_size) {
  while (dim % wg_size != 0 && wg_size > 1)
    wg_size--;

  return wg_size;
}
