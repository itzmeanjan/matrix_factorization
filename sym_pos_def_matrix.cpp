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

void add(sycl::queue &q, const float *mat_a, const float *mat_b,
         float *const mat_c, const uint dim, const uint wg_size) {
  sycl::buffer<float, 2> b_mat_a{mat_a, sycl::range<2>{dim, dim}};
  sycl::buffer<float, 2> b_mat_b{mat_b, sycl::range<2>{dim, dim}};
  sycl::buffer<float, 2> b_mat_c{mat_c, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 2, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        a_mat_a{b_mat_a, h};
    sycl::accessor<float, 2, sycl::access::mode::read,
                   sycl::access::target::global_buffer>
        a_mat_b{b_mat_b, h};
    sycl::accessor<float, 2, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        a_mat_c{b_mat_c, h, sycl::noinit};

    h.parallel_for<class kernelAdd>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint r = it.get_global_id(0);
          const uint c = it.get_global_id(1);

          a_mat_c[r][c] = a_mat_a[r][c] + a_mat_b[r][c];
        });
  });
  evt.wait();
}

void scalar_multiply(sycl::queue &q, float *const mat, const uint dim,
                     const uint wg_size, const float mult_factor) {
  sycl::buffer<float, 2> b_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 2, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        a_mat{b_mat, h};

    h.parallel_for<class kernelScalarMultiply>(
        sycl::nd_range<2>{sycl::range<2>{dim, dim}, sycl::range<2>{1, wg_size}},
        [=](sycl::nd_item<2> it) {
          const uint r = it.get_global_id(0);
          const uint c = it.get_global_id(1);

          a_mat[r][c] *= mult_factor;
        });
  });
  evt.wait();
}

void identity(sycl::queue &q, float *const mat, const uint dim,
              const uint wg_size) {
  memset(mat, 0, sizeof(float) * dim * dim);
  sycl::buffer<float, 2> b_mat{mat, sycl::range<2>{dim, dim}};

  auto evt = q.submit([&](sycl::handler &h) {
    sycl::accessor<float, 2, sycl::access::mode::write,
                   sycl::access::target::global_buffer>
        a_mat{b_mat, h};

    h.parallel_for<class kernelIdentity>(
        sycl::nd_range<1>{sycl::range<1>{dim}, sycl::range<1>{wg_size}},
        [=](sycl::nd_item<1> it) {
          const uint r = it.get_global_id(0);

          a_mat[r][r] = dim;
        });
  });
  evt.wait();
}
