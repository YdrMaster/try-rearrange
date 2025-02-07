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

extern "C" __global__ void fill_src(
    void *__restrict__ src,
    unsigned int n,
    unsigned int items_per_thread) {

    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int start_idx = thread_idx * items_per_thread;

    for (int i = 0; i < items_per_thread && start_idx + i < n; i++) {
        reinterpret_cast<char *>(src)[start_idx + i] = threadIdx.x % 256;
    }
}


template<class Tunit>
__device__ void internal2(
    Tunit *__restrict__ dst,
    Tidx const dst_sy,
    Tidx const dst_sx,
    Tunit const *__restrict__ src,
    Tidx const src_sy,
    Tidx const src_sx,
    Tidx const items_per_thread) {
    auto const
        x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y;
    for (int i = 0; i < items_per_thread; i++) {
        dst[(y * dst_sy + x * dst_sx + i * sizeof(Tunit)) / sizeof(Tunit)] = src[(y * src_sy + x * src_sx + i * sizeof(Tunit)) / sizeof(Tunit)];

        // dst[(y * dst_sy + x * dst_sx) / sizeof(Tunit)] = src[(y * src_sy + x * src_sx) / sizeof(Tunit)];
    }
}

extern "C" __global__ void rearrange2(
    void *__restrict__ dst,
    Tidx const dst_sy,
    Tidx const dst_sx,
    void const *__restrict__ src,
    Tidx const src_sy,
    Tidx const src_sx,
    uint8_t const unit,
    Tidx const items_per_thread) {

#define CASE(N, T)                                                                               \
    case N:                                                                                      \
        internal2((T *) dst, dst_sy, dst_sx, (T const *) src, src_sy, src_sx, items_per_thread); \
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
