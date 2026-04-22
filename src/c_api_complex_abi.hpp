/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <perflibs_sparse.h>

#include <complex.h>
#include <complex>

#if !defined(_WIN32)
// Some C++ SDKs declare the C99 complex functions in <complex.h> but do not
// make the names visible here. Declare the small subset we use explicitly so
// the bridge can stay on the standard C99 accessors instead of compiler-
// specific extensions.
extern "C" float crealf(float _Complex);
extern "C" float cimagf(float _Complex);
extern "C" double creal(double _Complex);
extern "C" double cimag(double _Complex);
#endif

namespace perflibs::sparse::c_api {

#if defined(_WIN32)

inline std::complex<float> to_cpp_scalar(_Fcomplex z) {
  return {z._Val[0], z._Val[1]};
}

inline std::complex<double> to_cpp_scalar(_Dcomplex z) {
  return {z._Val[0], z._Val[1]};
}

#else

inline std::complex<float> to_cpp_scalar(float _Complex z) {
  return {crealf(z), cimagf(z)};
}

inline std::complex<float> to_cpp_scalar(std::complex<float> z) { return z; }

inline std::complex<double> to_cpp_scalar(double _Complex z) {
  return {creal(z), cimag(z)};
}

inline std::complex<double> to_cpp_scalar(std::complex<double> z) { return z; }

#endif

} // namespace perflibs::sparse::c_api
