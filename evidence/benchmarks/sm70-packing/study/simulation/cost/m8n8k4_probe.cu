// Compilation probe only. Register inputs must follow PTX's four-MMA-per-warp
// m8n8k4 fragment map before this is used as a numerical matrix test.
#include <cuda_runtime.h>
extern "C" __global__ void m8n8k4_instruction_probe(
  const unsigned* a, const unsigned* b, float* out) {
  int lane=threadIdx.x&31;
  unsigned a0=a[2*lane],a1=a[2*lane+1],b0=b[2*lane],b1=b[2*lane+1];
  float d0,d1,d2,d3,d4,d5,d6,d7;
  asm volatile(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
    "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3),"=f"(d4),"=f"(d5),"=f"(d6),"=f"(d7)
    : "r"(a0),"r"(a1),"r"(b0),"r"(b1),
      "f"(0.0f),"f"(0.0f),"f"(0.0f),"f"(0.0f),
      "f"(0.0f),"f"(0.0f),"f"(0.0f),"f"(0.0f));
  out[8*lane+0]=d0; out[8*lane+1]=d1; out[8*lane+2]=d2; out[8*lane+3]=d3;
  out[8*lane+4]=d4; out[8*lane+5]=d5; out[8*lane+6]=d6; out[8*lane+7]=d7;
}
