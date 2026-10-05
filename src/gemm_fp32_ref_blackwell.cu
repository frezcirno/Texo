#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

// The compute mapping intentionally matches cuda/vec128. One 256-thread CTA
// owns a 128x128 output tile; its eight warps form a 2x4 array of 64x32 warp
// tiles, and every lane accumulates one contiguous 8x8 microtile.
constexpr int kWarpSize = 32;
constexpr int kBlockRows = 128;
constexpr int kBlockColumns = 128;
constexpr int kBlockInner = 32;
constexpr int kWarpRows = 64;
constexpr int kWarpColumns = 32;
constexpr int kThreadRows = 8;
constexpr int kThreadColumns = 8;
constexpr int kVectorWidth = 4;
constexpr int kThreadsPerBlock = 256;

constexpr int kSharedASkew = kBlockInner - kVectorWidth;
constexpr int kSharedAStride = kBlockRows + kSharedASkew;
constexpr int kATileElements = kBlockInner * kSharedAStride;
constexpr int kBTileElements = kBlockInner * kBlockColumns;
constexpr std::uint32_t kBTileBytes =
    static_cast<std::uint32_t>(kBTileElements * sizeof(float));

static_assert(kThreadsPerBlock == 8 * kWarpSize);
static_assert(kBlockRows % kWarpRows == 0);
static_assert(kBlockColumns % kWarpColumns == 0);
static_assert(
    (kBlockRows / kWarpRows) * (kBlockColumns / kWarpColumns) ==
    kThreadsPerBlock / kWarpSize);
static_assert(kWarpRows / kThreadRows == 8);
static_assert(kWarpColumns / kThreadColumns == 4);
static_assert(kBlockRows * kBlockInner / kVectorWidth ==
              4 * kThreadsPerBlock);
static_assert(kBTileElements / kVectorWidth == 4 * kThreadsPerBlock);
static_assert(kBTileBytes == 16 * 1024);

// ------------------------- TMA/mbarrier primitives -------------------------

__device__ __forceinline__ std::uint32_t shared_address(const void* pointer) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(std::uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const std::uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;"
        :
        : "r"(address)
        : "memory");
#else
    __trap();
#endif
}

__device__ __forceinline__ void mbarrier_init_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
    __trap();
#endif
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    std::uint64_t* barrier,
    std::uint32_t transaction_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const std::uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "_, [%0], %1;"
        :
        : "r"(address), "r"(transaction_bytes)
        : "memory");
#else
    __trap();
#endif
}

__device__ __forceinline__ void mbarrier_wait(
    const std::uint64_t* barrier,
    std::uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const std::uint32_t address = shared_address(barrier);
    constexpr std::uint32_t kSuspendHint = 0x989680;
    std::uint32_t ready = 0;
    do {
        asm volatile(
            "{\n\t"
            ".reg .pred complete;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "complete, [%1], %2, %3;\n\t"
            "selp.u32 %0, 1, 0, complete;\n\t"
            "}"
            : "=r"(ready)
            : "r"(address), "r"(phase), "r"(kSuspendHint)
            : "memory");
    } while (ready == 0);
#else
    __trap();
#endif
}

__device__ __forceinline__ void tma_load_b_2d(
    float* destination,
    const CUtensorMap* tensor_map,
    int column,
    int row,
    std::uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const std::uint32_t destination_address = shared_address(destination);
    const std::uint32_t barrier_address = shared_address(barrier);
    const std::uint64_t tensor_map_address =
        reinterpret_cast<std::uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(destination_address),
          "l"(tensor_map_address),
          "r"(column),
          "r"(row),
          "r"(barrier_address)
        : "memory");
#else
    __trap();
#endif
}

// A is transposed as [K][M] and each K row is shifted by one float4 per K
// chunk. During the transpose store, a warp covers eight K chunks and four
// rows: this skew maps those 32 lanes onto 32 distinct banks. The final K row
// needs seven float4 slots of padding. Compute groups remain contiguous and
// 16-byte aligned, so the two proven LDS.128 operations per thread still
// apply. The affine address also avoids the register pressure of an XOR map.
__device__ __forceinline__ int swizzled_a_index(int inner, int row) {
    const int chunk_skew = (inner / kVectorWidth) * kVectorWidth;
    return inner * kSharedAStride + row + chunk_skew;
}

// -------------------------- Fast-path SGEMM -------------------------------

__global__ __launch_bounds__(kThreadsPerBlock)
void sgemm_sm100_tma_vec128_kernel(
    const __grid_constant__ CUtensorMap b_map,
    const float* __restrict__ a,
    float* __restrict__ c,
    int n,
    int k) {
    // A is still loaded cooperatively, transposed, and bank-skewed. B is the
    // single TMA destination and remains in compute-native row-major order.
    // Both arrays are single-buffered deliberately: the CTA barrier at the end
    // of every K tile makes the next overwrite safe.
    __shared__ __align__(128) float shared_a[kATileElements];
    __shared__ __align__(128) float shared_b[kBlockInner][kBlockColumns];
    __shared__ __align__(8) std::uint64_t b_full_barrier;

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / kWarpSize;
    const int lane_id = thread_id % kWarpSize;

    constexpr int kWarpColumnsPerBlock = kBlockColumns / kWarpColumns;
    constexpr int kThreadColumnsPerWarp = kWarpColumns / kThreadColumns;
    const int warp_row = warp_id / kWarpColumnsPerBlock;
    const int warp_column = warp_id % kWarpColumnsPerBlock;
    const int thread_row = lane_id / kThreadColumnsPerWarp;
    const int thread_column = lane_id % kThreadColumnsPerWarp;

    const int thread_tile_row =
        warp_row * kWarpRows + thread_row * kThreadRows;
    const int thread_tile_column =
        warp_column * kWarpColumns + thread_column * kThreadColumns;
    const int block_row = blockIdx.y * kBlockRows;
    const int block_column = blockIdx.x * kBlockColumns;

    if (thread_id == 0) {
        mbarrier_init(&b_full_barrier);
        mbarrier_init_fence();
    }
    __syncthreads();

    float accumulators[kThreadRows][kThreadColumns] = {};

    for (int reduction_offset = 0, tile = 0;
         reduction_offset < n;
         reduction_offset += kBlockInner, ++tile) {
        // One elected thread arms the transaction barrier and starts the
        // complete 32x128 B transfer. All 256 threads then move A, so the TMA
        // engine can overlap B traffic with A's four float4 loads/thread.
        if (thread_id == 0) {
            mbarrier_arrive_expect_tx(&b_full_barrier, kBTileBytes);
            tma_load_b_2d(
                &shared_b[0][0],
                &b_map,
                block_column,
                reduction_offset,
                &b_full_barrier);
        }

        #pragma unroll
        for (int load = 0; load < 4; ++load) {
            const int vector_index = thread_id + load * kThreadsPerBlock;
            const int a_row = vector_index / (kBlockInner / kVectorWidth);
            const int a_column_vector =
                vector_index % (kBlockInner / kVectorWidth);
            const int a_column = a_column_vector * kVectorWidth;
            const float4 a_values = *reinterpret_cast<const float4*>(
                a + static_cast<std::size_t>(block_row + a_row) * n +
                reduction_offset + a_column);
            shared_a[swizzled_a_index(a_column + 0, a_row)] = a_values.x;
            shared_a[swizzled_a_index(a_column + 1, a_row)] = a_values.y;
            shared_a[swizzled_a_index(a_column + 2, a_row)] = a_values.z;
            shared_a[swizzled_a_index(a_column + 3, a_row)] = a_values.w;
        }

        // This CTA barrier serves two purposes: every A store is visible, and
        // thread 0 has issued the TMA before any lane tests completion. The
        // acquire wait then makes the TMA-written B tile visible. Reusing one
        // mbarrier toggles parity after each completed transaction.
        __syncthreads();
        mbarrier_wait(
            &b_full_barrier,
            static_cast<std::uint32_t>(tile & 1));

        // Keep the proven vec128 register blocking exactly: each inner step
        // issues two 128-bit loads for A and two for B, then 64 FP32 FMAs.
        #pragma unroll
        for (int inner = 0; inner < kBlockInner; ++inner) {
            float a_values[kThreadRows];
            float b_values[kThreadColumns];

            const float4 a_0 = *reinterpret_cast<const float4*>(
                &shared_a[swizzled_a_index(inner, thread_tile_row)]);
            const float4 a_1 = *reinterpret_cast<const float4*>(
                &shared_a[swizzled_a_index(
                    inner,
                    thread_tile_row + kVectorWidth)]);
            a_values[0] = a_0.x;
            a_values[1] = a_0.y;
            a_values[2] = a_0.z;
            a_values[3] = a_0.w;
            a_values[4] = a_1.x;
            a_values[5] = a_1.y;
            a_values[6] = a_1.z;
            a_values[7] = a_1.w;

            const float4 b_0 = *reinterpret_cast<const float4*>(
                &shared_b[inner][thread_tile_column]);
            const float4 b_1 = *reinterpret_cast<const float4*>(
                &shared_b[inner][thread_tile_column + kVectorWidth]);
            b_values[0] = b_0.x;
            b_values[1] = b_0.y;
            b_values[2] = b_0.z;
            b_values[3] = b_0.w;
            b_values[4] = b_1.x;
            b_values[5] = b_1.y;
            b_values[6] = b_1.z;
            b_values[7] = b_1.w;

            #pragma unroll
            for (int row = 0; row < kThreadRows; ++row) {
                #pragma unroll
                for (int column = 0; column < kThreadColumns; ++column) {
                    accumulators[row][column] = fmaf(
                        a_values[row],
                        b_values[column],
                        accumulators[row][column]);
                }
            }
        }

        // No lane may start overwriting the single A/B stage until every lane
        // has consumed it. This also separates successive parity generations.
        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < kThreadRows; ++row) {
        float* output =
            c + static_cast<std::size_t>(block_row + thread_tile_row + row) * k +
            block_column + thread_tile_column;
        *reinterpret_cast<float4*>(output) = make_float4(
            accumulators[row][0],
            accumulators[row][1],
            accumulators[row][2],
            accumulators[row][3]);
        *reinterpret_cast<float4*>(output + kVectorWidth) = make_float4(
            accumulators[row][4],
            accumulators[row][5],
            accumulators[row][6],
            accumulators[row][7]);
    }
}

// Tensor Maps require aligned bases and a 16-byte outer stride. The optimized
// kernel also relies on full 128x128x32 tiles and vectorized A/C accesses.
// Every other legal workload shape takes this independent guarded kernel.
__global__ void sgemm_sm100_tma_vec128_fallback_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || column >= k) {
        return;
    }

    const float* a_row = a + static_cast<std::size_t>(row) * n;
    const float* b_column = b + column;
    float accumulator = 0.0f;
    for (int inner = 0; inner < n; ++inner) {
        accumulator = fmaf(a_row[inner], *b_column, accumulator);
        b_column += k;
    }
    c[static_cast<std::size_t>(row) * k + column] = accumulator;
}

CUresult encode_b_tensor_map(
    CUtensorMap* tensor_map,
    const float* b,
    std::uint64_t n,
    std::uint64_t k) {
    constexpr std::uint32_t kRank = 2;
    const std::uint64_t global_dimensions[kRank] = {k, n};
    const std::uint64_t global_strides[kRank - 1] = {
        k * sizeof(float),
    };
    const std::uint32_t box_dimensions[kRank] = {
        kBlockColumns,
        kBlockInner,
    };
    const std::uint32_t element_strides[kRank] = {1, 1};

    return cuTensorMapEncodeTiled(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        kRank,
        const_cast<float*>(b),
        global_dimensions,
        global_strides,
        box_dimensions,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace

extern "C" void solve(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    constexpr std::uintptr_t kRequiredAlignment = sizeof(float4);
    const bool eligible =
        m % kBlockRows == 0 &&
        n % kBlockInner == 0 &&
        k % kBlockColumns == 0 &&
        reinterpret_cast<std::uintptr_t>(a) % kRequiredAlignment == 0 &&
        reinterpret_cast<std::uintptr_t>(b) % kRequiredAlignment == 0 &&
        reinterpret_cast<std::uintptr_t>(c) % kRequiredAlignment == 0;

    if (eligible) {
        CUtensorMap b_map = {};
        const CUresult result = encode_b_tensor_map(
            &b_map,
            b,
            static_cast<std::uint64_t>(n),
            static_cast<std::uint64_t>(k));
        if (result == CUDA_SUCCESS) {
            const dim3 blocks(k / kBlockColumns, m / kBlockRows);
            sgemm_sm100_tma_vec128_kernel<<<blocks, kThreadsPerBlock>>>(
                b_map,
                a,
                c,
                n,
                k);
            return;
        }
    }

    const dim3 threads(32, 8);
    const dim3 blocks((k + threads.x - 1) / threads.x,
                      (m + threads.y - 1) / threads.y);
    sgemm_sm100_tma_vec128_fallback_kernel<<<blocks, threads>>>(
        a,
        b,
        c,
        m,
        n,
        k);
}
