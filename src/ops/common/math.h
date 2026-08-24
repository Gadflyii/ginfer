#pragma once

#include <type_traits>

#if defined(__CUDACC__)
#    define GINFER_KERNEL_HD __host__ __device__
#else
#    define GINFER_KERNEL_HD
#endif

namespace ginfer::ops {

template <class T>
GINFER_KERNEL_HD constexpr T div_up(T x, T d) {
    static_assert(std::is_integral_v<T>, "div_up requires an integral type");
    return x / d + static_cast<T>(x % d != 0);
}

template <class T>
GINFER_KERNEL_HD constexpr T round_up(T x, T multiple) {
    static_assert(std::is_integral_v<T>, "round_up requires an integral type");
    return div_up(x, multiple) * multiple;
}

template <auto Alignment, class T>
GINFER_KERNEL_HD constexpr T align_up(T x) {
    static_assert(std::is_integral_v<T>, "align_up requires an integral type");
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
                  "align_up requires a power-of-two alignment");
    constexpr T a = static_cast<T>(Alignment);
    return (x + a - 1) & ~(a - 1);
}

template <int Bits>
GINFER_KERNEL_HD constexpr int sign_extend(int x) {
    static_assert(Bits > 0 && Bits < 32, "sign_extend supports bit widths in [1, 31]");
    constexpr int sign = 1 << (Bits - 1);
    return (x ^ sign) - sign;
}

} // namespace ginfer::ops

#undef GINFER_KERNEL_HD
