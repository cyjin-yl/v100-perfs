#include "fragment_mapping.hpp"
#include <iostream>
int main() {
  for(int lane=0;lane<32;++lane) {
    std::cout<<lane<<' '<<mma_group(lane)<<' '<<a_row(lane)<<' '<<b_col(lane);
    for(int i=0;i<8;++i)std::cout<<' '<<c_row(lane,i)<<' '<<c_col(lane,i);
    std::cout<<'\n';
  }
}
