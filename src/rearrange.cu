#include <cstdint>

using Tidx = int64_t;

template<class Tunit>
__device__ void internal(
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

extern "C" __global__ void rearrange(
    void *__restrict__ dst,
    Tidx const dst_sy,
    Tidx const dst_sx,
    void const *__restrict__ src,
    Tidx const src_sy,
    Tidx const src_sx,
    uint8_t const unit) {

#define CASE(N, T)                                                            \
    case N:                                                                   \
        internal((T *) dst, dst_sy, dst_sx, (T const *) src, src_sy, src_sx); \
        break

    switch (unit) {
        CASE(1, uint8_t);
        CASE(2, uint16_t);
        CASE(4, uint32_t);
        CASE(8, uint64_t);
        CASE(16, float4);
        CASE(32, double4);
        default:
            break;
    }

#undef CASE
}
