#pragma once

namespace p1 {

// Host API: takes host pointers, manages device memory internally.
void vector_add(const float* a, const float* b, float* c, int n);

// Device API: caller owns device memory.
void vector_add_device(const float* d_a, const float* d_b, float* d_c, int n);

}  // namespace p1
