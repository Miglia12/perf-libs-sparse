/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

namespace perflibs::sparse::statistics {
struct normal_distribution {
  double mean = 0;
  double stddev = 1;
  double n = 0;

  normal_distribution() = default;
  normal_distribution(double mean, double stddev, double n)
      : mean(mean), stddev(stddev), n(n) {}
};

double mean(const std::vector<double> &values);
double mean_incremental(normal_distribution in, double x);

double sample_variance(const std::vector<double> &values, double m);
double sample_variance(const std::vector<double> &values);

double sample_stddev(const std::vector<double> &values, double m);
double sample_stddev(const std::vector<double> &values);
double sample_stddev_incremental(normal_distribution in, double m, double x);

normal_distribution sample_normal(const std::vector<double> &values);
normal_distribution sample_normal_incremental(normal_distribution in, double x);

normal_distribution combine_normals(const normal_distribution &dist1,
                                    const normal_distribution &dist2);
normal_distribution
combine_normals(const std::vector<normal_distribution> &dists);

struct t_test_result {
  double t;
  double v;
  double v1_p; // probability if v=1
};

t_test_result welch_t_test(normal_distribution n1, normal_distribution n2);
t_test_result welch_t_test(const std::vector<double> &v1,
                           const std::vector<double> &v2);

normal_distribution operator*(const normal_distribution &a, double b);
normal_distribution operator*(double a, const normal_distribution &b);
normal_distribution operator/(const normal_distribution &a, double b);
normal_distribution operator/(double a, const normal_distribution &b);
} // namespace perflibs::sparse::statistics
