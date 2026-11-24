// Isolated reproduction of the arithmetic expression in upstream mma_impl.
// This is NOT a run of a PTX kernel, NOT a complete GPGPU-Sim functional test,
// and NOT V100 hardware evidence. It checks whether that expression is a
// suitable oracle for FP16-input / FP32-accumulation packing experiments.
#include "half.hpp"
#include <cmath>
#include <iostream>
int main() {
  using half_float::half;
  half a(1539.0f), b(3.0f), acc(0.0f);
  // Same intermediate type and expression as mma_impl matrix_d[i][j].f16.
  acc = acc + a*b;
  float one=static_cast<float>(acc);
  for(int k=1;k<16;++k) acc = acc+a*b;
  float sixteen=static_cast<float>(acc);
  std::cout << "{\"scope\":\"isolated upstream half-accumulator formula, not full simulator or GPU\","
            << "\"K1_result\":" << one << ",\"K1_exact_FP32_integer\":4617,"
            << "\"K16_is_inf\":" << (std::isinf(sixteen)?"true":"false")
            << ",\"K16_exact_FP32_integer\":73872}\n";
  return (one!=4617.0f && std::isinf(sixteen))?0:1;
}
