/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#include "util.hpp"

namespace perflibs::sparse {

const char *banner = " ** ";

// Print the string passed, inserting newlines before
// position = location for each line. Prepending the
// first line with a banner, and every other line with
// indentation.
void print_wrap(std::string str, bool include_banner) {

  // The indentation to use before each new line
  const std::string indent = "    ";
  // The location before which we wrap each line
  constexpr int64_t location = 80;
  // i tracks the position of the start of each line
  // we want this to be signed so that the test in
  // the while loop is false for one-line strings
  int64_t i = 0;

  auto insert = include_banner ? banner : indent;

  // Iterate while we're not on the last line, searching for spaces.
  // A newline for the last line is included in the printf.
  while (i < (int64_t)(str.length() + insert.length()) - location) {
    str.insert(i, insert);
    insert = indent; // after the first iter we always want to indent
    // search back from position 'loc' to find a space in which to insert '\n'
    auto loc = location + i;
    auto n = str.rfind(' ', loc);
    // the position of the start of the next line
    i = n + 1;
    if (n != std::string::npos) {
      str.at(n) = '\n';
    }
    // Get out if the search failed. This shouldn't
    // happen for reasonable values of location.
    else {
      break;
    }
  }

  if (i < (int64_t)str.length()) {
    str.insert(i, insert);
  }

  fprintf(stderr, "%s\n", str.c_str());
}

perflibs_status_t update_matrix_error(sp_error_t &error_handle,
                                      perflibs_int_t i) {
  error_handle.perflibs_error_type = PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
  error_handle.perflibs_error_code =
      1; // The matrix is always the first parameter

  std::string str0 = "The element requested to be updated in position ";
  std::string str1 = std::to_string(i);
  std::string str2 = " of the parameters passed to the update function does "
                     "not correspond to a non-zero "
                     "value in the sparse matrix. The update function can only "
                     "update existing values, it "
                     "cannot be used to introduce new non-zeros.";

  error_handle.err_msg = str0 + str1 + str2;

  return PERFLIBS_STATUS_INPUT_PARAMETER_ERROR;
}

} // namespace perflibs::sparse
