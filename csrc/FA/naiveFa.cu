


template <typename value_t>
__device__ void
mma_m16n8k16_f32_accum(
    float &d1, float &d2, float &d3, float &d4,
    uint32_t const &a1, uint32_t const &a2,
    uint32_t const &a3, uint32_t const &a4,

    uint32_t const &b1, uint32_t const &b2,

    float const &c1, float const &c2,
    float const &c3, float const &c4
) {

    static_assert(std::is_same_v<value_t, half> ||
                    std::is_same_v<value_t, nv_bfloat16>,
                    "value_t must bt either half or nv_bfloat16");

    if constexpr (std::is_same<value_t, nv_bfloat16>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
                        "{%0, %1, %2, %3},"
                        "{%4, %5, %6, %7},"
                        "{%8, %9},"
                        "{%10, %11, %12, %13};"
                        : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                        : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                          "f"(c1), "f"(c2), "f"(c3), "f"(c4));

    }else {
        // fp16

    }

}