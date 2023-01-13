> **Warning** I've stopped maintaining this project !

# matrix_factorization
Parallel Matrix Factorization on GPGPU

## Background

I've started writing matrix factorization implementations, targeting accelerators i.e. CPUs, GPGPUs, using SYCL DPC++

Some posts I wrote, should accompany them:

- [Cholesky Factorization](https://itzmeanjan.in/pages/parallel-cholesky-factorization.html)
- [Improved Cholesky Factorization](https://itzmeanjan.in/pages/improving-parallel-cholesky-factorization.html)

## Usage

- Make sure you've Intel oneAPI toolkit installed. I found [this](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt) helpful.
- Compile using


```bash
make
```

- Run using

```bash
./run
```

## Benchmark

### Cholesky Factorization

- On CPU_0

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Cholesky Factorization:

  dimension			      time			max deviation
32   x   32			         5 ms			7.62939e-06
64   x   64			         5 ms			3.05176e-05
128  x  128			        10 ms			0.00012207
256  x  256			        20 ms			0.000305176
512  x  512			        67 ms			 0.0012207
1024 x 1024			       370 ms			0.00610352
```

- On CPU_1

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Cholesky Factorization:

  dimension			      time			max deviation
32   x   32			         8 ms			1.14441e-05
64   x   64			         6 ms			3.8147e-05
128  x  128			        10 ms			9.15527e-05
256  x  256			        21 ms			0.000305176
512  x  512			        52 ms			0.000976562
1024 x 1024			       172 ms			0.00524902
```

- On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Cholesky Factorization:

  dimension			      time			max deviation
32   x   32			         2 ms			7.62939e-06
64   x   64			         3 ms			3.8147e-05
128  x  128			         8 ms			9.15527e-05
256  x  256			        21 ms			0.000427246
512  x  512			        78 ms			0.000976562
1024 x 1024			       305 ms			0.00439453
```
