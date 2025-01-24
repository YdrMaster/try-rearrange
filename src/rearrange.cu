#include <cstdint>

using Tunit = uint64_t;
using Tidx = int64_t;

extern "C" __global__ void rearrange(
    Tunit *__restrict__ dst,
    Tidx const dst_sy,
    Tidx const dst_sx,
    Tunit const *__restrict__ src,
    Tidx const src_sy,
    Tidx const src_sx) {
    auto const
        x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y;
    dst[(y * dst_sy + x * dst_sx) / sizeof(Tunit)] = src[(y * src_sy + x * src_sx) / sizeof(Tunit)];
}
