#pragma once

// Per-step trace of the parallel CSR solve, compiled only with
// SPARSE_LEVEL_TRACE: for every thread, the clock at arrival at each barrier
// and at release from it. Everything else (step durations, busy time, barrier
// latency) is computed outside the library from these two reads.

#include "int.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace perflibs::sparse {

// The generic counter: one system-wide clock, readable from user space on
// every AArch64 core. isb: every earlier instruction, stores included, has
// completed before the counter is read. "memory": the compiler may not move
// loads or stores across the read.
inline uint64_t level_trace_clock() {
  uint64_t ticks;
  asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(ticks) : : "memory");
  return ticks;
}

inline uint64_t level_trace_freq() {
  uint64_t hz;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(hz));
  return hz;
}

// One row per thread: [region start, arrival 0, release 0, arrival 1, ...].
// Rows are padded to a multiple of 32 entries (256 B) so that no two threads
// write to the same cache line.
struct level_trace_t {
  std::vector<uint64_t> ticks;
  size_t row_len = 0;
  int nthreads = 0;
  perflibs_int_t steps = 0;
  int call = 0;
};
inline level_trace_t level_trace;

} // namespace perflibs::sparse
