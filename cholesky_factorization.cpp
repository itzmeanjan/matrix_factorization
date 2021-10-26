#include "cholesky_factorization.hpp"
#include "sym_pos_def_matrix.hpp"
#include "utils.hpp"

using namespace sycl;

// Read: https://cseweb.ucsd.edu//~hskim/files/cse260.pdf
int64_t cholesky(queue &q, const float *mat_in, float *const mat_out,
                 const uint dim, const uint wg_size) {
  memcpy(mat_out, mat_in, sizeof(float) * dim * dim);

  buffer<float, 2> b_mat_out{mat_out, range<2>{dim, dim}};

  q.submit([&](handler &h) {
    accessor<float, 2, access::mode::write, access::target::global_buffer>
        a_mat_out{b_mat_out, h};

    h.parallel_for<class kernelZeroLower>(
        nd_range<2>{range<2>{dim, dim}, range<2>{1, wg_size}},
        [=](nd_item<2> it) {
          const uint i = it.get_global_id(0);
          const uint j = it.get_global_id(1);

          if (i > j) {
            a_mat_out[i][j] = 0.f;
          }
        });
  });

  std::chrono::_V2::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (uint k = 0; k < dim; k++) {

    q.submit([&](handler &h) {
      accessor<float, 2, access::mode::read_write,
               access::target::global_buffer>
          a_mat_out{b_mat_out, h, range<2>{k + 1, 1}, id<2>{0, k}};

      h.parallel_for<class kernelPivotCalc>(
          nd_range<1>{range<1>{k},
                      range<1>{compute_work_group_size(k, wg_size)}},
          [=](nd_item<1> it) {
            const uint i = it.get_global_id(0);

            auto ref = sycl::ONEAPI::atomic_ref<
                float, sycl::ONEAPI::memory_order::relaxed,
                sycl::ONEAPI::memory_scope::work_group,
                access::address_space::global_device_space>(a_mat_out[k][0]);
            ref.fetch_sub(sycl::pow(a_mat_out[i][0], 2.f));
          });
    });

    q.submit([&](handler &h) {
      accessor<float, 2, access::mode::read_write,
               access::target::global_buffer>
          a_mat_out{b_mat_out, h, range<2>{1, 1}, id<2>{k, k}};

      h.single_task([=]() { a_mat_out[0][0] = sycl::sqrt(a_mat_out[0][0]); });
    });

    q.submit([&](handler &h) {
      accessor<float, 2, access::mode::read_write,
               access::target::global_buffer>
          a_mat_out{b_mat_out, h};

      const uint dim_ = dim - (k + 1);
      h.parallel_for<class kernelRowCalc>(
          nd_range<1>{range<1>{dim_},
                      range<1>{compute_work_group_size(dim_, wg_size)},
                      id<1>{k + 1}},
          [=](nd_item<1> it) {
            const uint i = it.get_global_id(0);

            float sum = 0.f;
            for (uint j = 0; j < k; j++) {
              sum += a_mat_out[j][k] * a_mat_out[j][i];
            }

            a_mat_out[k][i] -= sum;
            a_mat_out[k][i] /= a_mat_out[k][k];
          });
    });
  }

  q.wait();

  std::chrono::_V2::steady_clock::time_point end =
      std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  const uint N = 1 << 10;
  const uint B = 1 << 5;

  for (uint dim = B; dim <= N; dim <<= 1) {
    uint size = sizeof(float) * dim * dim;

    float *mat_in = (float *)malloc(size);
    float *mat_out = (float *)malloc(size);
    float *mat_fac = (float *)malloc(size);
    float *mat_fac_ = (float *)malloc(size);

    random_matrix(mat_in, dim);
    memset(mat_out, 0, size);
    memset(mat_fac, 0, size);

    int64_t ts_0 =
        gen_symmetric_positive_definite_matrix(q, mat_in, mat_out, dim, B);
    int64_t ts_1 = cholesky(q, mat_out, mat_fac, dim, B);

    memcpy(mat_fac_, mat_fac, size);
    transpose(q, mat_fac_, dim, B);

    float max_dev = 0.f;
    for (uint i = 0; i < dim; i++) {
      for (uint j = 0; j < dim; j++) {
        float sum = 0.f;
        for (uint k = 0; k < dim; k++) {
          sum += mat_fac_[i * dim + k] * mat_fac[k * dim + j];
        }
        max_dev = std::max(max_dev, std::abs(mat_out[i * dim + j] - sum));
      }
    }

    std::cout << dim << "x" << dim << "\t\t\t" << ts_1 << "ms\t\t\t" << max_dev
              << std::endl;

    std::free(mat_in);
    std::free(mat_out);
    std::free(mat_fac);
    std::free(mat_fac_);
  }

  return 0;
}
