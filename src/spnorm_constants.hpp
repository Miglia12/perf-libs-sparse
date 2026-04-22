/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <cmath>
#include <limits>

namespace perflibs::sparse {

// Base of the number system used in fp representation
template <typename T> constexpr T radix = std::numeric_limits<T>::radix;

// Number of digits of mantissa in fp representation
template <typename T> constexpr T t = std::numeric_limits<T>::digits;

// Min and max exponents in fp representation
template <typename T>
constexpr T exp_min = std::numeric_limits<T>::min_exponent;
template <typename T>
constexpr T exp_max = std::numeric_limits<T>::max_exponent;

// Parameters for safe scaling, defined in
// https://dl.acm.org/doi/pdf/10.1145/3061665 t_big and s_small are large
// positive numbers t_small and s_big are small positive numbers
template <typename T>
const T t_big = pow(radix<T>, std::floor((exp_max<T> - t<T> + 1) / 2));

template <typename T>
const T t_small = pow(radix<T>, std::ceil((exp_min<T> - 1) / 2));

template <typename T>
const T s_big = pow(radix<T>, -std::ceil((exp_max<T> - t<T> + 1) / 2));

template <typename T>
const T s_small = pow(radix<T>, -std::floor((exp_min<T> - t<T>) / 2));

} // namespace perflibs::sparse
