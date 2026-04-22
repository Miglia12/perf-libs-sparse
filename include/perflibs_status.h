/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#ifndef PERFLIBS_STATUS_H
#define PERFLIBS_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ENUMs */

typedef enum perflibs_status {
  PERFLIBS_STATUS_SUCCESS = 0,
  PERFLIBS_STATUS_INPUT_PARAMETER_ERROR = 1,
  PERFLIBS_STATUS_EXECUTION_FAILURE = 2,
} perflibs_status_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // PERFLIBS_STATUS_H
