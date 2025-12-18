#pragma once

#include <DataTypes/IDataType.h>

namespace DB::Compression::FFOR
{
constexpr UInt16 DEFAULT_VALUES = 1024;

template <typename T>
concept UnsignedInteger = std::is_integral_v<T> && std::is_unsigned_v<T> && !std::is_same_v<T, bool>;

template <UnsignedInteger T>
consteval UInt16 get_bit_width()
{
    return sizeof(T) * 8u;
}

template <UnsignedInteger T, UInt16 values = DEFAULT_VALUES>
consteval UInt16 get_iterations()
{
    constexpr UInt16 width = get_bit_width<T>();
    return values / width;
}

template <UnsignedInteger T, UInt16 bits, UInt16 step, UInt16 values = DEFAULT_VALUES>
ALWAYS_INLINE void bitPackStep(const T * __restrict in, T * __restrict out, const T base, const UInt16 index, T & state)
{
    constexpr UInt16 width = get_bit_width<T>();
    static_assert(bits <= width);
    static_assert(step <= width);

    constexpr UInt16 iterations = get_iterations<T, values>();

    if constexpr (bits > 0 && bits < width)
    {
        constexpr UInt16 shift = (step * bits) % width;

        constexpr T mask = (T{1} << bits) - 1;

        // Check if a new output word is needed
        if constexpr (step > 0 && shift < bits)
        {
            constexpr UInt16 out_offset = iterations * (((step - 1) * bits) / width);
            out[out_offset + index] = state;

            if constexpr (step >= width)
                return;

            if constexpr (shift > 0)
            {
                constexpr UInt16 in_prev_offset = iterations * (step - 1);
                constexpr UInt16 prev_shit = bits - shift;
                T v = in[in_prev_offset + index] - base;
                v &= mask;
                v >>= prev_shit;
                state = v;
            }
            else
                state = 0;
        }
        else if constexpr (step == 0)
            state = 0;

        if constexpr (step < width)
        {
            constexpr UInt16 in_offset = iterations * step;
            T v = in[in_offset + index] - base;
            v &= mask;
            v <<= shift;
            state |= v;
        }
    }
    else if constexpr (bits == width && step < width)
    {
        constexpr UInt16 offset = iterations * step;
        out[offset + index] = in[offset + index] - base;
    }
}

template <UnsignedInteger T, UInt16 bits, UInt16 values = DEFAULT_VALUES>
void bitPack(const T * __restrict in, T * __restrict out, const T base)
{
    constexpr UInt16 width = get_bit_width<T>();
    static_assert(bits <= width);

    if constexpr (bits == 0)
        return;

    constexpr UInt16 iterations = get_iterations<T, values>();

    for (UInt16 i = 0; i < iterations; ++i)
    {
        [&]<UInt16... step>(std::integer_sequence<UInt16, step ...>) ALWAYS_INLINE
        {
            T state = 0;
            ((bitPackStep<T, bits, step, values>(in, out, base, i, state)), ...);
        }(std::make_integer_sequence<UInt16, width + 1>{});
    }
}

template <UnsignedInteger T, unsigned bits, unsigned step, unsigned values = DEFAULT_VALUES>
ALWAYS_INLINE void bitUnpackStep(const T * __restrict in, T * __restrict out, T base, unsigned i, T& reg) {
    constexpr UInt16 width = get_bit_width<T>();
    static_assert(bits <= width);
    static_assert(step <= width);

    constexpr unsigned iterations = get_iterations<T, values>();

    if constexpr (bits == 0)
        out[i + iterations * step] = base;
    else if constexpr (bits == width)
    {
        constexpr UInt16 offset = iterations * step;
        out[offset + i] = in[offset + i] + base;
    }
    else {
        constexpr unsigned bit_shift = (step * bits) % width;
        constexpr unsigned bits_to_full = width - bit_shift;
        constexpr unsigned in_shift = iterations * ((step + 1) * bits / width);

        T tmp;

        if constexpr (bits_to_full < bits) {
            constexpr unsigned shift = bits - bits_to_full;
            constexpr T mask1 = (T{1} << bits_to_full) - 1;
            constexpr T mask2 = (T{1} << shift) - 1;

            tmp = (reg >> bit_shift) & mask1;
            if constexpr (step < width - 1)
            {
                reg = in[i + in_shift];
                tmp |= (reg & mask2) << bits_to_full;
            }
        } else {
            if constexpr (bit_shift == 0)
            {
                if constexpr (step < width - 1)
                    reg = in[i + in_shift];
                else
                    reg = 0;
            }

            constexpr T mask = (T{1} << bits) - 1;
            tmp = (reg >> bit_shift) & mask;
        }

        tmp += base;
        out[i + iterations * step] = tmp;
    }
}

template <UnsignedInteger T, unsigned bits, unsigned values = DEFAULT_VALUES>
void bitUnpack(const T * __restrict in, T * __restrict out, T base) {
    constexpr UInt16 width = get_bit_width<T>();
    static_assert(bits <= width);

    constexpr unsigned iterations = get_iterations<T, values>();

    for (unsigned i = 0; i < iterations; ++i) {
        [&]<unsigned... step>(std::integer_sequence<unsigned, step ...>) __attribute__((always_inline)) {
            T reg = 0;
            ((bitUnpackStep<T, bits, step, values>(in, out, base, i, reg)), ...);
        }(std::make_integer_sequence<unsigned, width>{});
    }
}

template <unsigned values = DEFAULT_VALUES>
void bitPack(const uint64_t* __restrict in, uint64_t* __restrict out, unsigned bits, uint64_t base) {
    assert(bits <= 64);
    switch (bits) {
        case 0:  bitPack<uint64_t, 0,  values>(in, out, base); break;
        case 1:  bitPack<uint64_t, 1,  values>(in, out, base); break;
        case 2:  bitPack<uint64_t, 2,  values>(in, out, base); break;
        case 3:  bitPack<uint64_t, 3,  values>(in, out, base); break;
        case 4:  bitPack<uint64_t, 4,  values>(in, out, base); break;
        case 5:  bitPack<uint64_t, 5,  values>(in, out, base); break;
        case 6:  bitPack<uint64_t, 6,  values>(in, out, base); break;
        case 7:  bitPack<uint64_t, 7,  values>(in, out, base); break;
        case 8:  bitPack<uint64_t, 8,  values>(in, out, base); break;
        case 9:  bitPack<uint64_t, 9,  values>(in, out, base); break;
        case 10: bitPack<uint64_t, 10, values>(in, out, base); break;
        case 11: bitPack<uint64_t, 11, values>(in, out, base); break;
        case 12: bitPack<uint64_t, 12, values>(in, out, base); break;
        case 13: bitPack<uint64_t, 13, values>(in, out, base); break;
        case 14: bitPack<uint64_t, 14, values>(in, out, base); break;
        case 15: bitPack<uint64_t, 15, values>(in, out, base); break;
        case 16: bitPack<uint64_t, 16, values>(in, out, base); break;
        case 17: bitPack<uint64_t, 17, values>(in, out, base); break;
        case 18: bitPack<uint64_t, 18, values>(in, out, base); break;
        case 19: bitPack<uint64_t, 19, values>(in, out, base); break;
        case 20: bitPack<uint64_t, 20, values>(in, out, base); break;
        case 21: bitPack<uint64_t, 21, values>(in, out, base); break;
        case 22: bitPack<uint64_t, 22, values>(in, out, base); break;
        case 23: bitPack<uint64_t, 23, values>(in, out, base); break;
        case 24: bitPack<uint64_t, 24, values>(in, out, base); break;
        case 25: bitPack<uint64_t, 25, values>(in, out, base); break;
        case 26: bitPack<uint64_t, 26, values>(in, out, base); break;
        case 27: bitPack<uint64_t, 27, values>(in, out, base); break;
        case 28: bitPack<uint64_t, 28, values>(in, out, base); break;
        case 29: bitPack<uint64_t, 29, values>(in, out, base); break;
        case 30: bitPack<uint64_t, 30, values>(in, out, base); break;
        case 31: bitPack<uint64_t, 31, values>(in, out, base); break;
        case 32: bitPack<uint64_t, 32, values>(in, out, base); break;
        case 33: bitPack<uint64_t, 33, values>(in, out, base); break;
        case 34: bitPack<uint64_t, 34, values>(in, out, base); break;
        case 35: bitPack<uint64_t, 35, values>(in, out, base); break;
        case 36: bitPack<uint64_t, 36, values>(in, out, base); break;
        case 37: bitPack<uint64_t, 37, values>(in, out, base); break;
        case 38: bitPack<uint64_t, 38, values>(in, out, base); break;
        case 39: bitPack<uint64_t, 39, values>(in, out, base); break;
        case 40: bitPack<uint64_t, 40, values>(in, out, base); break;
        case 41: bitPack<uint64_t, 41, values>(in, out, base); break;
        case 42: bitPack<uint64_t, 42, values>(in, out, base); break;
        case 43: bitPack<uint64_t, 43, values>(in, out, base); break;
        case 44: bitPack<uint64_t, 44, values>(in, out, base); break;
        case 45: bitPack<uint64_t, 45, values>(in, out, base); break;
        case 46: bitPack<uint64_t, 46, values>(in, out, base); break;
        case 47: bitPack<uint64_t, 47, values>(in, out, base); break;
        case 48: bitPack<uint64_t, 48, values>(in, out, base); break;
        case 49: bitPack<uint64_t, 49, values>(in, out, base); break;
        case 50: bitPack<uint64_t, 50, values>(in, out, base); break;
        case 51: bitPack<uint64_t, 51, values>(in, out, base); break;
        case 52: bitPack<uint64_t, 52, values>(in, out, base); break;
        case 53: bitPack<uint64_t, 53, values>(in, out, base); break;
        case 54: bitPack<uint64_t, 54, values>(in, out, base); break;
        case 55: bitPack<uint64_t, 55, values>(in, out, base); break;
        case 56: bitPack<uint64_t, 56, values>(in, out, base); break;
        case 57: bitPack<uint64_t, 57, values>(in, out, base); break;
        case 58: bitPack<uint64_t, 58, values>(in, out, base); break;
        case 59: bitPack<uint64_t, 59, values>(in, out, base); break;
        case 60: bitPack<uint64_t, 60, values>(in, out, base); break;
        case 61: bitPack<uint64_t, 61, values>(in, out, base); break;
        case 62: bitPack<uint64_t, 62, values>(in, out, base); break;
        case 63: bitPack<uint64_t, 63, values>(in, out, base); break;
        case 64: bitPack<uint64_t, 64, values>(in, out, base); break;
    }
}

template <unsigned values = DEFAULT_VALUES>
void bitUnpack(const uint64_t* __restrict in, uint64_t* __restrict out, unsigned bits, uint64_t base) {
    assert(bits <= 64);
    switch (bits) {
        case 0:  bitUnpack<uint64_t, 0,  values>(in, out, base); break;
        case 1:  bitUnpack<uint64_t, 1,  values>(in, out, base); break;
        case 2:  bitUnpack<uint64_t, 2,  values>(in, out, base); break;
        case 3:  bitUnpack<uint64_t, 3,  values>(in, out, base); break;
        case 4:  bitUnpack<uint64_t, 4,  values>(in, out, base); break;
        case 5:  bitUnpack<uint64_t, 5,  values>(in, out, base); break;
        case 6:  bitUnpack<uint64_t, 6,  values>(in, out, base); break;
        case 7:  bitUnpack<uint64_t, 7,  values>(in, out, base); break;
        case 8:  bitUnpack<uint64_t, 8,  values>(in, out, base); break;
        case 9:  bitUnpack<uint64_t, 9,  values>(in, out, base); break;
        case 10: bitUnpack<uint64_t, 10, values>(in, out, base); break;
        case 11: bitUnpack<uint64_t, 11, values>(in, out, base); break;
        case 12: bitUnpack<uint64_t, 12, values>(in, out, base); break;
        case 13: bitUnpack<uint64_t, 13, values>(in, out, base); break;
        case 14: bitUnpack<uint64_t, 14, values>(in, out, base); break;
        case 15: bitUnpack<uint64_t, 15, values>(in, out, base); break;
        case 16: bitUnpack<uint64_t, 16, values>(in, out, base); break;
        case 17: bitUnpack<uint64_t, 17, values>(in, out, base); break;
        case 18: bitUnpack<uint64_t, 18, values>(in, out, base); break;
        case 19: bitUnpack<uint64_t, 19, values>(in, out, base); break;
        case 20: bitUnpack<uint64_t, 20, values>(in, out, base); break;
        case 21: bitUnpack<uint64_t, 21, values>(in, out, base); break;
        case 22: bitUnpack<uint64_t, 22, values>(in, out, base); break;
        case 23: bitUnpack<uint64_t, 23, values>(in, out, base); break;
        case 24: bitUnpack<uint64_t, 24, values>(in, out, base); break;
        case 25: bitUnpack<uint64_t, 25, values>(in, out, base); break;
        case 26: bitUnpack<uint64_t, 26, values>(in, out, base); break;
        case 27: bitUnpack<uint64_t, 27, values>(in, out, base); break;
        case 28: bitUnpack<uint64_t, 28, values>(in, out, base); break;
        case 29: bitUnpack<uint64_t, 29, values>(in, out, base); break;
        case 30: bitUnpack<uint64_t, 30, values>(in, out, base); break;
        case 31: bitUnpack<uint64_t, 31, values>(in, out, base); break;
        case 32: bitUnpack<uint64_t, 32, values>(in, out, base); break;
        case 33: bitUnpack<uint64_t, 33, values>(in, out, base); break;
        case 34: bitUnpack<uint64_t, 34, values>(in, out, base); break;
        case 35: bitUnpack<uint64_t, 35, values>(in, out, base); break;
        case 36: bitUnpack<uint64_t, 36, values>(in, out, base); break;
        case 37: bitUnpack<uint64_t, 37, values>(in, out, base); break;
        case 38: bitUnpack<uint64_t, 38, values>(in, out, base); break;
        case 39: bitUnpack<uint64_t, 39, values>(in, out, base); break;
        case 40: bitUnpack<uint64_t, 40, values>(in, out, base); break;
        case 41: bitUnpack<uint64_t, 41, values>(in, out, base); break;
        case 42: bitUnpack<uint64_t, 42, values>(in, out, base); break;
        case 43: bitUnpack<uint64_t, 43, values>(in, out, base); break;
        case 44: bitUnpack<uint64_t, 44, values>(in, out, base); break;
        case 45: bitUnpack<uint64_t, 45, values>(in, out, base); break;
        case 46: bitUnpack<uint64_t, 46, values>(in, out, base); break;
        case 47: bitUnpack<uint64_t, 47, values>(in, out, base); break;
        case 48: bitUnpack<uint64_t, 48, values>(in, out, base); break;
        case 49: bitUnpack<uint64_t, 49, values>(in, out, base); break;
        case 50: bitUnpack<uint64_t, 50, values>(in, out, base); break;
        case 51: bitUnpack<uint64_t, 51, values>(in, out, base); break;
        case 52: bitUnpack<uint64_t, 52, values>(in, out, base); break;
        case 53: bitUnpack<uint64_t, 53, values>(in, out, base); break;
        case 54: bitUnpack<uint64_t, 54, values>(in, out, base); break;
        case 55: bitUnpack<uint64_t, 55, values>(in, out, base); break;
        case 56: bitUnpack<uint64_t, 56, values>(in, out, base); break;
        case 57: bitUnpack<uint64_t, 57, values>(in, out, base); break;
        case 58: bitUnpack<uint64_t, 58, values>(in, out, base); break;
        case 59: bitUnpack<uint64_t, 59, values>(in, out, base); break;
        case 60: bitUnpack<uint64_t, 60, values>(in, out, base); break;
        case 61: bitUnpack<uint64_t, 61, values>(in, out, base); break;
        case 62: bitUnpack<uint64_t, 62, values>(in, out, base); break;
        case 63: bitUnpack<uint64_t, 63, values>(in, out, base); break;
        case 64: bitUnpack<uint64_t, 64, values>(in, out, base); break;
    }
}

template <UInt16 values = DEFAULT_VALUES>
UInt16 calculateBitpackedBytes(UInt16 bits)
{
    assert(bits <= 64);
    return bits * DEFAULT_VALUES / 8;
}
}
