#include <CL/sycl.hpp>

void transpose(sycl::queue &q, float *const mat, const uint dim,
               const uint wg_size);

void add(sycl::queue &q, const float *mat_a, const float *mat_b,
         float *const mat_c, const uint dim, const uint wg_size);

void scalar_multiply(sycl::queue &q, float *const mat, const uint dim,
                     const uint wg_size, const float mult_factor);
