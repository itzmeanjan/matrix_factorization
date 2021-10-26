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

32x32			5ms			1.14441e-05
64x64			4ms			3.8147e-05
128x128			9ms			0.00012207
256x256			17ms			0.000305176
512x512			72ms			0.00109863
1024x1024			413ms			0.0045166
```

- On CPU_1

```bash
running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

32x32			9ms			1.14441e-05
64x64			6ms			5.34058e-05
128x128			10ms			0.000106812
256x256			22ms			0.000274658
512x512			52ms			0.00134277
1024x1024			168ms			0.00476074
```

- On GPU

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

32x32			2ms			7.62939e-06
64x64			3ms			3.8147e-05
128x128			7ms			0.00012207
256x256			21ms			0.000274658
512x512			75ms			0.00134277
1024x1024			305ms			0.00476074
```
