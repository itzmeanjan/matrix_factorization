#include <CL/sycl.hpp>

void transpose(sycl::queue &q, float *const mat, const uint dim,
               const uint wg_size);
