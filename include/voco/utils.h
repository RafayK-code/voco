#pragma once

#define VOCO_ENUM_CLASS_FLAG_OPERATORS(T)                                                           \
    inline constexpr T operator|(T a, T b) { return T(int(a) | int(b)); }                   \
    inline constexpr T operator&(T a, T b) { return T(int(a) & int(b)); }                   \
    inline constexpr T operator~(T a) { return T(~int(a)); }                           \
    inline constexpr bool operator!(T a) { return int(a) == 0; }                          \
    inline constexpr bool operator==(T a, int b) { return int(a) == b; }                          \
    inline constexpr bool operator!=(T a, int b) { return int(a) != b; }