#include "sym_pos_def_matrix.hpp"

void transpose(sycl::queue &q, float *const mat, const uint dim,
               const uint wg_size) {
  sycl::buffer<float, 2> b_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 2, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        a_mat{b_mat, h};

    h.parallel_for<class kernelTranspose>(sycl::nd_range<1>{dim, wg_size},
                                          [=](sycl::nd_item<1> it) {
                                            const uint r = it.get_global_id(0);

                                            for (uint c = 0; c < r; c++) {
                                              const float tmp = a_mat[r][c];
                                              a_mat[r][c] = a_mat[c][r];
                                              a_mat[c][r] = tmp;
                                            }
                                          });
  });
  evt.wait();
}
