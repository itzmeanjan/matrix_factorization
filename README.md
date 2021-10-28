# matrix_factorization
Parallel Matrix Factorization on GPGPU

## Background

I've started writing matrix factorization implementations, targeting accelerators i.e. CPUs, GPGPUs, using SYCL DPC++

Some posts I wrote, should accompany them:

- [Cholesky Factorization](https://itzmeanjan.in/pages/parallel-cholesky-factorization.html)

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

32   x   32			         5 ms			1.14441e-05
64   x   64			         4 ms			4.57764e-05
128  x  128			         9 ms			9.15527e-05
256  x  256			        17 ms			0.000427246
512  x  512			        71 ms			0.00115967
1024 x 1024			       409 ms			0.00610352
```

- On CPU_1

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Cholesky Factorization:

32   x   32			         8 ms			1.14441e-05
64   x   64			         5 ms			2.28882e-05
128  x  128			        10 ms			9.15527e-05
256  x  256			        23 ms			0.000274658
512  x  512			        55 ms			0.00128174
1024 x 1024			       179 ms			0.00488281
```

- On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Cholesky Factorization:

32   x   32			         2 ms			7.62939e-06
64   x   64			         3 ms			3.8147e-05
128  x  128			         8 ms			9.15527e-05
256  x  256			        21 ms			0.000427246
512  x  512			        78 ms			0.000976562
1024 x 1024			       305 ms			0.00439453
```
