#include "cute/arch/mma_sm80.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

template<typename Spec, bool IsGemm>
__global__ void cute_hgemm(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {
    using namespace cute;

    using X = Underscore;
    using T = typename Spec::T;
    using TiledMMA = typename Spec::TiledMMA;

    constexpr int kTileM = Spec::kTileM;
    constexpr int kTileN = Spec::kTileN;
    constexpr int kTileK = Spec::kTileK;

    int tid = threadIdx.x;
    Tensor mA = make_tensor(make_gmem_ptr((T*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  // (M, K)
    Tensor mB = make_tensor(make_gmem_ptr((T*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr((T*)Cptr), make_shape(m, n), make_stride(n ,Int<1>{}));  // (M, N)

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});    
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});  // (kTileM, kTileK)
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});  // (kTileN, kTileK)
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});  // (kTileM, kTileN)

    // Equivalent to:
    // Tensor gA = local_tile(mA, make_shape(Int<kTileM>{}, Int<kTileK>{}), make_coord(0, 0));  // (kTileM, kTileK)
    // Tensor gB = local_tile(mB, make_shape(Int<kTileN>{}, Int<kTileK>{}), make_coord(0, 0));  // (kTileN, kTileK)
    // Tensor gC = local_tile(mC, make_shape(Int<kTileM>{}, Int<kTileN>{}), make_coord(0, 0));  // (kTileM, kTileN)

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    Tensor tCrA = thr_mma.partition_fragment_A(gA); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(gB); // (MMA, MMA_N, MMA_K)
    Tensor tCrC = thr_mma.partition_fragment_C(gC); // (MMA, MMA_M, MMA_N)

    auto copy_atom = AutoVectorizingCopy{};

    copy(copy_atom, tCgA, tCrA);
    copy(copy_atom, tCgB, tCrB);

    if constexpr (IsGemm) {
        clear(tCrC);
    }else {
        copy(copy_atom, tCgC, tCrC);
    }

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    copy(copy_atom, tCrC, tCgC);

    // if (thread0()) {
    //   print_latex(tiled_mma); printf("\n");
    //   print(tCgA); printf("\n");
    //   print(tCgB); printf("\n");
    //   print(tCgC); printf("\n");
    //   print(tCrA); printf("\n");
    //   print(tCrB); printf("\n");
    //   print(tCrC); printf("\n");
    // }
}

namespace spec {
    using namespace cute;
    template<typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8> struct KernelSpec{
        using T = T_;
        static constexpr int kTileM = kTileM_;
        static constexpr int kTileN = kTileN_;
        static constexpr int kTileK = kTileK_;

        using MMA_op = SM80_16x8x8_F16F16F16F16_TN;
        //using MMA_op = SM70_8x8x4_F16F16F16F16_TN;
        using TiledMMA = decltype(make_tiled_mma(MMA_op{}));

        static constexpr int kThreadNum = size(TiledMMA{});
        static constexpr int kShmSize = 0;
    };
}

