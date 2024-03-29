#include <CL/sycl.hpp>

int64_t cholesky(sycl::queue &q, const float *mat_in, float *const mat_out,
                 const uint dim, const uint wg_size);
