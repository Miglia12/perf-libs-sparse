/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "sell_c_sigma.hpp"

namespace perflibs::sparse {

template<typename T>
void spmv_scs_opt(const perflibs_scs<T> &scs, const T *x, T *y, T alpha, T beta) {

	switch (scs.C) {
		%for C in range(2, 18, 2):
		case ${C}:
			spmv_scs<T, ${C}>(scs, x, y, alpha, beta);
			break;
		%endfor
	}
}
template void spmv_scs_opt<std::complex<float>>(const perflibs_scs<std::complex<float>> &scs, const std::complex<float> *x, std::complex<float> *y, std::complex<float> alpha, std::complex<float> beta);
template void spmv_scs_opt<std::complex<double>>(const perflibs_scs<std::complex<double>> &scs, const std::complex<double> *x, std::complex<double> *y, std::complex<double> alpha, std::complex<double> beta);

%if target_os == 'windows_arm64ec':
// When building for Arm64EC we use the C++ SCS implementation for float and double, instead of the
// versions which use assembly kernels
template void spmv_scs_opt<float>(const perflibs_scs<float> &scs, const float *x, float *y, float alpha, float beta);
template void spmv_scs_opt<double>(const perflibs_scs<double> &scs, const double *x, double *y, double alpha, double beta);
%endif
}
