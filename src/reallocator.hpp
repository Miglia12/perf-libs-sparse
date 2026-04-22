/*
 * SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
 * affiliates <open-source-office@arm.com></text>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

// This is a narrow internal allocation helper used where the project wants
// realloc-style growth for trivially relocatable storage rather than the fuller
// standard allocator/container stack.

namespace perflibs::sparse {

/** A derivative of the standard allocator interface which offers reallocation
 * of previously allocated memory. std::realloc is used for all reallocations,
 * and like std::realloc an exact number of bytes must be specified for
 * reallocation.
 */
class reallocator {
public:
  using pointer = void *;
  using const_pointer = const pointer;
  using size_type = std::size_t;

  constexpr reallocator() noexcept = default;

  constexpr reallocator(const reallocator &other) noexcept = default;

  ~reallocator() = default;

  pointer reallocate(const_pointer ptr, size_type new_size) {
    if (new_size >
        static_cast<size_type>(std::numeric_limits<std::ptrdiff_t>::max())) {
      return nullptr;
    }
    return static_cast<pointer>(std::realloc(ptr, new_size));
  }

  void deallocate(const_pointer ptr) { std::free(ptr); }
};

} // namespace perflibs::sparse
