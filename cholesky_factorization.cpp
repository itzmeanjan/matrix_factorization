#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>

using namespace sycl;

const uint N = 1 << 10;
const uint B = 1 << 5;
const float MULT_FACTOR = .5f;

// Read: https://cseweb.ucsd.edu//~hskim/files/cse260.pdf
int64_t cholesky(queue &q, const float *mat_in, float *const mat_out) {
  memcpy(mat_out, mat_in, sizeof(float) * N * N);

  buffer<float, 2> b_mat_out{mat_out, range<2>{N, N}};

  std::chrono::_V2::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (uint k = 0; k < N; k++) {

    if (k > 0) {
      q.submit([&](handler &h) {
        accessor<float, 2, access::mode::read_write,
                 access::target::global_buffer>
            a_mat_out{b_mat_out, h};

        h.parallel_for<class kernelPivotCalc>(
            nd_range<1>{range<1>{N}, range<1>{B}}, [=](nd_item<1> it) {
              const uint i = it.get_global_id(0);

              if (i >= 0 && i < k) {
                auto ref = sycl::ONEAPI::atomic_ref<
                    float, sycl::ONEAPI::memory_order::relaxed,
                    sycl::ONEAPI::memory_scope::work_group,
                    access::address_space::global_device_space>(
                    a_mat_out[k][k]);
                ref.fetch_sub(sycl::pow(a_mat_out[i][k], 2.f));
              }
            });
      });
    }

    {
      host_accessor<float, 2, access::mode::read_write> h_mat_out{b_mat_out};
      h_mat_out[k][k] = sycl::sqrt(h_mat_out[k][k]);
    }

    q.submit([&](handler &h) {
      accessor<float, 2, access::mode::read_write,
               access::target::global_buffer>
          a_mat_out{b_mat_out, h};

      h.parallel_for<class kernelRowCalc>(
          nd_range<1>{range<1>{N}, range<1>{B}}, [=](nd_item<1> it) {
            const uint i = it.get_global_id(0);

            if (i > k && i < N) {
              float sum = 0.f;
              for (uint j = 0; j < k; j++) {
                sum += a_mat_out[j][k] * a_mat_out[j][i];
              }
              a_mat_out[k][i] -= sum;
              a_mat_out[k][i] /= a_mat_out[k][k];
            }
          });
    });
  }

  auto evt = q.submit([&](handler &h) {
    accessor<float, 2, access::mode::write, access::target::global_buffer>
        a_mat_out{b_mat_out, h};

    h.parallel_for<class kernelZeroLower>(
        nd_range<2>{range<2>{N, N}, range<2>{1, B}}, [=](nd_item<2> it) {
          const uint i = it.get_global_id(0);
          const uint j = it.get_global_id(1);

          if (i > j) {
            a_mat_out[i][j] = 0.f;
          }
        });
  });
  evt.wait();

  std::chrono::_V2::steady_clock::time_point end =
      std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void identity(queue &q, float *const mat) {
  memset(mat, 0, sizeof(float) * N * N);
  buffer<float, 2> b_mat{mat, range<2>{N, N}};

  auto evt = q.submit([&](handler &h) {
    accessor<float, 2, access::mode::write, access::target::global_buffer>
        a_mat{b_mat, h};

    h.parallel_for<class kernelIdentity>(nd_range<1>{range<1>{N}, range<1>{B}},
                                         [=](nd_item<1> it) {
                                           const uint r = it.get_global_id(0);

                                           a_mat[r][r] = N;
                                         });
  });
  evt.wait();
}

void scalar_multiply(queue &q, float *const mat) {
  buffer<float, 2> b_mat{mat, range<2>{N, N}};

  auto evt = q.submit([&](handler &h) {
    accessor<float, 2, access::mode::read_write, access::target::global_buffer>
        a_mat{b_mat, h};

    h.parallel_for<class kernelScalarMultiply>(
        nd_range<2>{range<2>{N, N}, range<2>{1, B}}, [=](nd_item<2> it) {
          const uint r = it.get_global_id(0);
          const uint c = it.get_global_id(1);

          a_mat[r][c] *= MULT_FACTOR;
        });
  });
  evt.wait();
}

void add(queue &q, const float *mat_a, const float *mat_b, float *const mat_c) {
  buffer<float, 2> b_mat_a{mat_a, range<2>{N, N}};
  buffer<float, 2> b_mat_b{mat_b, range<2>{N, N}};
  buffer<float, 2> b_mat_c{mat_c, range<2>{N, N}};

  auto evt = q.submit([&](handler &h) {
    accessor<float, 2, access::mode::read, access::target::global_buffer>
        a_mat_a{b_mat_a, h};
    accessor<float, 2, access::mode::read, access::target::global_buffer>
        a_mat_b{b_mat_b, h};
    accessor<float, 2, access::mode::write, access::target::global_buffer>
        a_mat_c{b_mat_c, h, noinit};

    h.parallel_for<class kernelAdd>(
        nd_range<2>{range<2>{N, N}, range<2>{1, B}}, [=](nd_item<2> it) {
          const uint r = it.get_global_id(0);
          const uint c = it.get_global_id(1);

          a_mat_c[r][c] = a_mat_a[r][c] + a_mat_b[r][c];
        });
  });
  evt.wait();
}

void transpose(queue &q, float *const mat) {
  buffer<float, 2> b_mat{mat, range<2>{N, N}};

  auto evt = q.submit([&](handler &h) {
    accessor<float, 2, access::mode::read_write, access::target::global_buffer>
        a_mat{b_mat, h};

    h.parallel_for<class kernelTranspose>(nd_range<1>{N, B},
                                          [=](nd_item<1> it) {
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

int64_t gen_symmetric_positive_definite_matrix(queue &q, const float *in_mat,
                                               float *const out_mat) {
  const uint size = sizeof(float) * N * N;

  float *mat_b = (float *)malloc(size);
  float *mat_c = (float *)malloc(size);
  float *mat_d = (float *)malloc(size);

  memcpy(mat_b, in_mat, size);
  memset(mat_c, 0, size);
  memset(mat_d, 0, size);
  memset(out_mat, 0, size);

  std::chrono::_V2::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  transpose(q, mat_b);
  add(q, in_mat, mat_b, mat_c);
  scalar_multiply(q, mat_c);
  identity(q, mat_d);
  add(q, mat_c, mat_d, out_mat);

  std::chrono::_V2::steady_clock::time_point end =
      std::chrono::steady_clock::now();

  std::free(mat_b);
  std::free(mat_c);
  std::free(mat_d);

  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void random_matrix(float *const matrix) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      matrix[i * N + j] = dis(gen);
    }
  }
}

void show(const float *mat) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      std::cout << mat[i * N + j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << std::endl;

  uint size = sizeof(float) * N * N;

  float *mat_in = (float *)malloc(size);
  float *mat_out = (float *)malloc(size);
  float *mat_fac = (float *)malloc(size);
  float *mat_fac_ = (float *)malloc(size);

  random_matrix(mat_in);
  memset(mat_out, 0, size);
  memset(mat_fac, 0, size);

  int64_t ts_0 = gen_symmetric_positive_definite_matrix(q, mat_in, mat_out);
  int64_t ts_1 = cholesky(q, mat_out, mat_fac);

  memcpy(mat_fac_, mat_fac, size);
  transpose(q, mat_fac_);

  float max_dev = 0.f;
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      float sum = 0.f;
      for (uint k = 0; k < N; k++) {
        sum += mat_fac_[i * N + k] * mat_fac[k * N + j];
      }
      max_dev = std::max(max_dev, std::abs(mat_out[i * N + j] - sum));
    }
  }

  std::cout << "\nrandom symmetric positive definite matrix generated, in "
            << ts_0 << " ms\n"
            << "cholesky factorization, in " << ts_1 << " ms\t|\tmax deviation "
            << max_dev << std::endl;

  std::free(mat_in);
  std::free(mat_out);
  std::free(mat_fac);
  std::free(mat_fac_);

  return 0;
}
