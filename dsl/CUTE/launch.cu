#include <torch/extension.h>
#include "cute_hgemm.cuh"

#define CHECK_TORCH_TENSOR_DTYPE(T, DTYPE)                                                                            \
  do {                                                                                                                \
    if ((T).options().dtype() != (DTYPE)) {                                                                           \
      std::cerr << "Tensor dtype mismatch! Expected: " << (DTYPE) << ", but got: " << (T).options().dtype() << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define CHECK_TORCH_TENSOR_SHAPE(T, M, N)                                                                             \
  do {                                                                                                                \
    auto actual_shape = (T).sizes();                                                                                  \
    if (actual_shape != torch::IntArrayRef({M, N})) {                                                                 \
      std::cerr << "Tensor shape mismatch! Expected: " << torch::IntArrayRef({M, N}) << ", but got: " << actual_shape \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                            \
  [&] {                                                                                                               \
    if (COND) {                                                                                                       \
      constexpr static bool CONST_NAME = true;                                                                        \
      return __VA_ARGS__();                                                                                           \
    } else {                                                                                                          \
      constexpr static bool CONST_NAME = false;                                                                       \
      return __VA_ARGS__();                                                                                           \
    }                                                                                                                 \
  }()


template <typename ComputeType, typename AccType = ComputeType>
torch::Tensor run_cute_hgemm(const torch::Tensor &a, const torch::Tensor &b, std::optional<torch::Tensor> &_c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  const int M = 16;
  const int N = 8;
  const int K = 8;

  auto torch_compute_type = [] {
    if constexpr (std::is_same_v<ComputeType, cute::half_t>) {
      return torch::kHalf; 
    }
    throw std::runtime_error("Unsupported ComputeType!");
  }();

  auto torch_acc_type = [] {
    if constexpr (std::is_same_v<AccType, cute::half_t>) {
      return torch::kHalf; 
    }
    throw std::runtime_error("Unsupported ComputeType!");
  }();

  torch::Tensor c;
  bool is_gemm;

  if(!_c.has_value()) {
      auto options = torch::TensorOptions().dtype(torch_acc_type).device(torch::kCUDA);
      c = torch::empty({M, N}, options);
      is_gemm = true;
  }else {
      c = _c.value();
      is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K);
  CHECK_TORCH_TENSOR_SHAPE(b, N, K);
  CHECK_TORCH_TENSOR_SHAPE(c, M, N);

  using Spec = spec::KernelSpec<ComputeType, M, N, K>;

  //cute::print(typename Spec::TiledMMA{});

  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
    block.z, grid.x, grid.y, grid.z, shm_size);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  BOOL_SWITCH(is_gemm, IsGemm, [&]{
    cudaEventRecord(start, stream);
    cute_hgemm<Spec, IsGemm><<<grid, block, shm_size, stream>>>(reinterpret_cast<AccType*>(c.data_ptr()), reinterpret_cast<ComputeType*>(a.data_ptr()), reinterpret_cast<ComputeType*>(b.data_ptr()), M, N, K);
    cudaEventRecord(stop, stream);
  });

  cudaDeviceSynchronize();

  auto error = cudaGetLastError();
  if(error != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) + "error code: " + std::to_string(error) + ")");
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %.3f ms\n", milliseconds); 

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cute_hgemm", &(run_cute_hgemm<cute::half_t>), "Run a single 8x8x4 MMA operation.");
}