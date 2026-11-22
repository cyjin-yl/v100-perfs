#pragma once
#ifdef __CUDACC__
#define MAP_HD __host__ __device__
#else
#define MAP_HD
#endif
// PTX ISA 7.4 section 9.7.13.4.1, m8n8k4 .f16, A row / B col.
// Four 8x8 groups per warp are placed at the four quadrants of a 16x16 tile.
MAP_HD constexpr int mma_group(int lane) { return (lane >> 2) & 3; }
MAP_HD constexpr int quadrant_row(int lane) { return (mma_group(lane) >> 1) * 8; }
MAP_HD constexpr int quadrant_col(int lane) { return (mma_group(lane) & 1) * 8; }
MAP_HD constexpr int input_row_or_col(int lane) { return (lane & 3) + ((lane >> 4) * 4); }
MAP_HD constexpr int a_row(int lane) { return quadrant_row(lane) + input_row_or_col(lane); }
MAP_HD constexpr int b_col(int lane) { return quadrant_col(lane) + input_row_or_col(lane); }
MAP_HD constexpr int c_local_row(int lane,int i) { return (lane & 1) + (i & 2) + ((lane >> 4) * 4); }
MAP_HD constexpr int c_local_col(int lane,int i) { return (i & 4) + (lane & 2) + (i & 1); }
MAP_HD constexpr int c_row(int lane,int i) { return quadrant_row(lane) + c_local_row(lane,i); }
MAP_HD constexpr int c_col(int lane,int i) { return quadrant_col(lane) + c_local_col(lane,i); }
#undef MAP_HD
