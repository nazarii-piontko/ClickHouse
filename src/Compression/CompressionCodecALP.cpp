#include <Compression/CompressionFactory.h>
#include <Compression/CompressionInfo.h>
#include <Compression/ICompressionCodec.h>

#include <Parsers/IAST.h>
#include <DataTypes/IDataType.h>

#include <base/unaligned.h>

#include <IO/WriteHelpers.h>
#include <IO/BitHelpers.h>

#include <array>

#include "ALP/ffor.h"

namespace DB
{
/**
 * ALP (Adaptive Lossless floating-Point) compression codec.
 *
 * This implementation is based on the ALP paper (https://ir.cwi.nl/pub/33334) and implements the main ALP variant for Float32 and Float64.
 * It encodes floating-point values as scaled integers (plus a small set of exceptions), then applies Frame-of-Reference + bit-packing.
 *
 * Overall Stream Layout
 *   [ALP codec header]  2 bytes, written once per compressed column
 *   [ALP block 0]
 *   [ALP block 1]
 *   ...
 *   [ALP block N-1]
 *
 * ALP codec header (4 bytes):
 *   - meta_byte (1 byte):
 *     - bits 0-3: codec version (currently 1)
 *     - bit 4:    variant flag (0 = ALP, 1 = ALP_RD; only 0 supported)
 *     - bits 5-7: reserved (must be 0)
 *   - float_width (1 byte):
 *     - 4 or 8 bytes (Float32 / Float64)
 *   - block_float_count (2 bytes):
 *     - number of floats per block (UInt16), currently fixed to 1024
 *
 * The input column is split into blocks of up to ALP_MAX_BLOCK_FLOAT_COUNT values (1024).
 * Each block is encoded independently and can be either compressed or left raw, depending on the estimated gain.
 *
 * Core Idea: Decimal-Based Integerization
 * The ALP paper observes that most stored doubles originate as decimals. For a block, we try to represent each float value as an integer:
 *   1) multiplier(e, f) = 10^e * 10^(-f)
 *   2) d = (Int64) round(v * multiplier(e, f))
 * where:
 *   - v is the original floating-point value
 *   - d is the encoded integer
 *   - e controls the decimal scaling up
 *   - f controls how many trailing decimal zeros we effectively “cut off” again
 * A value is considered exactly encodable if:
 *   decodeValue(encodeValue(v, e, f), e, f) == v
 * Encodable values become part of the integer stream; non-encodable values become exceptions stored verbatim.
 * Encodable eligibility is based on bit-exact equality, not epsilon-based closeness, because the codec is lossless.
 *
 * Two-Level Sampling to Select (e,f)
 * ALP’s adaptivity over a block is driven by a two-level sampling scheme:
 *   1) Global pre-sampling over the entire column:
 *     - Take ALP_PARAMS_ESTIMATION_SAMPLES disjoint sub-samples from the column (each up to ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS values).
 *     - For each sub-sample, brute-force over all valid (e,f) pairs with 0 <= e < |EXPONENTS| and 0 <= f <= e.
 *     - For each (e,f), estimate the encoded size and keep track of the best (e,f) for that sub-sample.
 *     - The result is a small, global candidate set of (e,f) pairs (up to 5) that are likely good for this column.
 *   2) Per-block refinement
 *     - For each block of up to 1024 values, take a local sample
 *     - Evaluate only the global candidate (e,f) list for that sample to choose the best (e,f) for that block.
 *
 * Per-Block Encoding Schema (Compressed Case)
 *   - 1 byte: exponent (e) (UInt8)
 *   - 1 byte: fraction (f) (UInt8)
 *   - 2 bytes: exceptions count (UInt16)
 *   - 1 byte:  FOR bit-width (UInt8)
 *   - 8 bytes: FOR base value (Int64)
 *   - Compressed block payload:
 *     * Bit-packed values: (encoded_value - FOR_base) for each successfully encoded float, using bit-width bits per value.
 *     * Exceptions (for values that couldn't round-trip losslessly):
 *       - UInt16 index (position in source block - offset)
 *       - raw value (float or double)
 *
 * Per-Block Encoding Schema (Uncompressed Case)
 *   - 1 byte - equal to 255
 *   - Raw numbers for the block
 *
 * Notes:
 *   - This codec implements the ALP variant only.
 *     ALP_RD (front-bits-based encoder for “real doubles”) is not implemented.
 *   - Supported types: 4 and 8 bytes floating point.
 *   - The scheme is fully lossless: all non-exception values are proven round-trip encodable; all others are stored as raw exceptions.
 */
class CompressionCodecALP final : public ICompressionCodec
{
public:
    explicit CompressionCodecALP(UInt8 float_width_);
    uint8_t getMethodByte() const override;
    void updateHash(SipHash & hash) const override;
protected:
    UInt32 doCompressData(const char * source, UInt32 source_size, char * dest) const override;
    void doDecompressData(const char * source, UInt32 source_size, char * dest, UInt32 uncompressed_size) const override;
    UInt32 getMaxCompressedDataSize(UInt32 uncompressed_size) const override;
    bool isCompression() const override { return true; }
    bool isGenericCompression() const override { return false; }
    bool isFloatingPointTimeSeriesCodec() const override { return true; }
    String getDescription() const override;
private:
    UInt8 float_width;
};

namespace ErrorCodes
{
    extern const int CANNOT_COMPRESS;
    extern const int CANNOT_DECOMPRESS;
    extern const int BAD_ARGUMENTS;
    extern const int ILLEGAL_SYNTAX_FOR_CODEC_TYPE;
}

namespace
{
constexpr UInt8 ALP_CODEC_VERSION = 1;

constexpr UInt32 ALP_CODEC_HEADER_SIZE = 2 * sizeof(UInt8) + sizeof(UInt16);
constexpr UInt32 ALP_BLOCK_HEADER_SIZE = 3 * sizeof(UInt8) + sizeof(UInt16) + sizeof(Int64);
constexpr UInt32 ALP_UNENCODED_BLOCK_HEADER_SIZE = sizeof(UInt8);
constexpr UInt8 ALP_UNENCODED_BLOCK_EXPONENT = 255;

constexpr UInt32 ALP_MAX_BLOCK_FLOAT_COUNT = 1024;

constexpr UInt32 ALP_PARAMS_ESTIMATION_SAMPLES = 8;
constexpr UInt32 ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS = 32;
constexpr UInt32 ALP_PARAMS_ESTIMATION_CANDIDATES = 5;

template <typename T>
concept FLOAT = std::is_same_v<T, Float32> || std::is_same_v<T, Float64>;

template <FLOAT T>
struct ALPFloatTraits;

template <FLOAT T, UInt32 exponent_count, bool inverse>
constexpr std::array<T, exponent_count> generatePowersOf10()
{
    std::array<T, exponent_count> arr{};
    for (UInt64 i = 0, v = 1; i < exponent_count; ++i, v *= 10)
        arr[i] = inverse ? static_cast<T>(1) / static_cast<T>(v) : static_cast<T>(v);
    return arr;
}

template<>
struct ALPFloatTraits<Float64>
{
    // Covers scale factors 10^0 through 10^17; 10^18 exceeds Int64 max
    static constexpr UInt8 EXPONENT_COUNT = 18;

    static constexpr std::array<Float64, EXPONENT_COUNT> EXPONENTS = generatePowersOf10<Float64, EXPONENT_COUNT, false>();
    static constexpr std::array<Float64, EXPONENT_COUNT> FRACTIONS = generatePowersOf10<Float64, EXPONENT_COUNT, true>();

    static constexpr Float64 UPPER = 922337203685477478.0;
    static constexpr Float64 LOWER = -922337203685477478.0;

    static constexpr Float64 ROUND_MAGIC = 6755399441055744.0; // 2^51 + 2^52
};

template<>
struct ALPFloatTraits<Float32>
{
    // Covers scale factors 10^0 through 10^9; 10 instead of 18 due to Float32 precision limits.
    static constexpr UInt8 EXPONENT_COUNT = 10;

    static constexpr std::array<Float32, EXPONENT_COUNT> EXPONENTS = generatePowersOf10<Float32, EXPONENT_COUNT, false>();
    static constexpr std::array<Float32, EXPONENT_COUNT> FRACTIONS = generatePowersOf10<Float32, EXPONENT_COUNT, true>();

    static constexpr Float32 UPPER = 922337203685477478.0f;
    static constexpr Float32 LOWER = -922337203685477478.0f;

    static constexpr Float32 ROUND_MAGIC = 12582912.0; // 2^22 + 2^23
};

template<FLOAT T>
struct ALPFloatUtils
{
    static Int64 encodeValue(T value, UInt8 exponent, UInt8 fraction)
    {
        T value_enc = value * ALPFloatTraits<T>::EXPONENTS[exponent] * ALPFloatTraits<T>::FRACTIONS[fraction];

        const bool invalid = std::isinf(value_enc) ||
            std::isnan(value_enc) ||
                value_enc < ALPFloatTraits<T>::LOWER || value_enc > ALPFloatTraits<T>::UPPER ||
                    (value_enc == static_cast<T>(0.0) && std::signbit(value_enc));

        if (unlikely(invalid))
            return static_cast<Int64>(ALPFloatTraits<T>::UPPER);

        // Fast rounding to integer by adding and subtracting a large constant (IEEE-754 mantissa limit)
        value_enc = value_enc + ALPFloatTraits<T>::ROUND_MAGIC - ALPFloatTraits<T>::ROUND_MAGIC;
        return static_cast<Int64>(value_enc);
    }

    static T decodeValue(Int64 value, UInt8 exponent, UInt8 fraction)
    {
        T value_float = static_cast<T>(value);
        T value_dec = value_float * ALPFloatTraits<T>::EXPONENTS[fraction] * ALPFloatTraits<T>::FRACTIONS[exponent];
        return value_dec;
    }
};

template <FLOAT T>
class ALPCodecEncoder
{
public:
    explicit ALPCodecEncoder() = default;

    UInt32 encode(const char * source, const UInt32 source_size, char * dest)
    {
        if (source_size % sizeof(T) != 0)
            throw Exception(ErrorCodes::CANNOT_COMPRESS, "Cannot compress with ALP codec, data size {} is not aligned to {}", source_size, sizeof(T));

        UInt32 float_count = source_size / sizeof(T);

        estimateParamCandidates(source, float_count);

        const char * dest_start = dest;
        while (float_count >= ALP_MAX_BLOCK_FLOAT_COUNT)
        {
            dest = encodeBlock(source, ALP_MAX_BLOCK_FLOAT_COUNT, dest);

            source += ALP_MAX_BLOCK_FLOAT_COUNT * sizeof(T);
            float_count -= ALP_MAX_BLOCK_FLOAT_COUNT;
        }

        if (float_count > 0)
            dest = encodeBlock(source, static_cast<UInt16>(float_count), dest);

        return static_cast<UInt32>(dest - dest_start);
    }

private:
    struct EncodingParams
    {
        UInt8 exponent;
        UInt8 fraction;
    };

    struct EncodingException
    {
        UInt16 index;
        T value;
    };

    std::vector<EncodingParams> param_candidates;

    EncodingParams block_params;

    alignas(64) Int64 block_encoded_floats[ALP_MAX_BLOCK_FLOAT_COUNT];
    alignas(64) UInt64 block_bitpacked[ALP_MAX_BLOCK_FLOAT_COUNT];
    UInt32 block_encoded_float_count;
    UInt32 block_bitpacked_bytes;
    UInt8 block_bit_width;
    Int64 block_frame_of_reference;

    EncodingException block_exceptions[ALP_MAX_BLOCK_FLOAT_COUNT];
    UInt32 block_exceptions_count;

    char * encodeBlock(const char * source, const UInt16 float_count, char * dest)
    {
        encodeBlockToBuffer(source, float_count);

        const size_t total_encoded_bytes = ALP_BLOCK_HEADER_SIZE + block_bitpacked_bytes + block_exceptions_count * (sizeof(UInt16) + sizeof(T));
        const size_t total_unencoded_size = ALP_UNENCODED_BLOCK_HEADER_SIZE + float_count * sizeof(T);
        if (total_encoded_bytes >= total_unencoded_size) // No compression gain
            return writeUnencoded(source, float_count, dest);

        // Exponent and Fraction Indices
        *dest++ = block_params.exponent;
        *dest++ = block_params.fraction;

        // Exception Count
        unalignedStoreLittleEndian<UInt16>(dest, static_cast<UInt16>(block_exceptions_count));
        dest += sizeof(UInt16);

        // Encoding Bit-Width
        *dest++ = block_bit_width;

        // Frame of Reference Value
        unalignedStoreLittleEndian<Int64>(dest, block_frame_of_reference);
        dest += sizeof(Int64);

        // Write Encoded Values
        memcpy(dest, block_bitpacked, block_bitpacked_bytes);
        dest += block_bitpacked_bytes;

        // Write Exceptions
        for (UInt32 i = 0; i < block_exceptions_count; ++i)
        {
            const EncodingException & exception = block_exceptions[i];
            unalignedStoreLittleEndian<UInt16>(dest, exception.index);
            unalignedStoreLittleEndian<T>(dest + sizeof(UInt16), exception.value);
            dest += sizeof(UInt16) + sizeof(T);
        }

        return dest;
    }

    const char * encodeBlockToBuffer(const char * source, const UInt16 float_count)
    {
        block_params = selectBlockParams(source, float_count);

        block_encoded_float_count = 0;
        block_exceptions_count = 0;

        Int64 min = std::numeric_limits<Int64>::max();
        Int64 max = std::numeric_limits<Int64>::min();

        for (UInt16 i = 0; i < float_count; ++i, source += sizeof(T))
        {
            const T value = unalignedLoadLittleEndian<T>(source);
            const Int64 value_enc = ALPFloatUtils<T>::encodeValue(value, block_params.exponent, block_params.fraction);
            const T value_dec = ALPFloatUtils<T>::decodeValue(value_enc, block_params.exponent, block_params.fraction);

            block_encoded_floats[block_encoded_float_count++] = value_enc;

            if (likely(value == value_dec))
            {
                min = std::min(value_enc, min);
                max = std::max(value_enc, max);
            }
            else
                block_exceptions[block_exceptions_count++] = {i, value};
        }

        block_frame_of_reference = min;
        block_bit_width = calculateBitWidth(min, max);
        block_bitpacked_bytes = ALP::FFOR::calculateBitpackedSize(block_bit_width);

        // Fill exceptions positions with min value to simplify FOR encoding
        for (UInt32 i = 0; i < block_exceptions_count; ++i)
            block_encoded_floats[block_exceptions[i].index] = block_frame_of_reference;

        // Fill remaining positions with min value (if any)
        std::fill(block_encoded_floats + block_encoded_float_count, block_encoded_floats + ALP_MAX_BLOCK_FLOAT_COUNT, block_frame_of_reference);

        const auto *ffor_in = reinterpret_cast<const UInt64 *>(block_encoded_floats);
        UInt64 * ffor_out = block_bitpacked;
        const auto *ffor_base_p = reinterpret_cast<const UInt64 *>(&block_frame_of_reference);
        ALP::FFOR::ffor(ffor_in, ffor_out, block_bit_width, ffor_base_p);

        return source;
    }

    static char * writeUnencoded(const char * source, const UInt16 float_count, char * dest)
    {
        *dest++ = ALP_UNENCODED_BLOCK_EXPONENT; // Unencoded block marker

        const size_t block_size = float_count * sizeof(T);
        memcpy(dest, source, block_size);
        dest += block_size;

        return dest;
    }

    EncodingParams selectBlockParams(const char * source, const UInt32 float_count)
    {
        assert(param_candidates.size() > 0);
        if (param_candidates.size() == 1)
            return param_candidates[0];

        // Sample up to ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS values from the block for local parameter estimation.
        // Evenly select sample values across the block and copy them into a temporary buffer for evaluation.
        T sample[ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS];
        const UInt32 sample_count = std::min<UInt32>(float_count, ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS);
        const UInt32 sample_step = std::max(float_count / sample_count, 1u);
        for (UInt32 i = 0; i < sample_count; ++i)
            sample[i] = unalignedLoadLittleEndian<T>(&source[i * sample_step * sizeof(T)]);

        EncodingParams best_params = {0, 0};
        size_t best_size = std::numeric_limits<size_t>::max();
        bool is_prev_worse = false;

        for (const auto & params : param_candidates)
        {
            const size_t estimated_size = estimateEncodedSize(sample, sample_count, params);
            if (estimated_size < best_size)
            {
                best_size = estimated_size;
                best_params = params;
                is_prev_worse = false;
            }
            else if (estimated_size == best_size)
                is_prev_worse = false;
            else
            {
                if (is_prev_worse)
                    break; // Early stop if two consecutive candidates are worse
                is_prev_worse = true;
            }
        }

        return best_params;
    }

    void estimateParamCandidates(const char * source, const UInt32 float_count)
    {
        struct Estimation
        {
            EncodingParams params;
            UInt32 occurred_times;
        };
        std::unordered_map<UInt16, Estimation> estimations_map;

        // Take ALP_PARAMS_ESTIMATION_SAMPLES samples from the entire column for global parameter estimation.
        // Evenly select sample positions across the column.
        const UInt32 sample_step = float_count > ALP_PARAMS_ESTIMATION_SAMPLES * ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS
            ? (float_count - ALP_PARAMS_ESTIMATION_SAMPLES * ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS) / ALP_PARAMS_ESTIMATION_SAMPLES
            : ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS;

        T sample[ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS];

        // For each sample, brute-force over all valid (exponent, fraction) pairs to find the best parameters for that sample.
        for (UInt32 i = 0; i < ALP_PARAMS_ESTIMATION_SAMPLES; ++i)
        {
            const UInt32 sample_start_index = i * sample_step;
            if (sample_start_index >= float_count)
                break;

            const char * sample_pos = source + sample_start_index * sizeof(T);
            const UInt16 sample_float_count = std::min<UInt16>(ALP_PARAMS_ESTIMATION_SAMPLE_FLOATS, float_count - sample_start_index);

            for (UInt32 j = 0; j < sample_float_count; ++j, sample_pos += sizeof(T))
                sample[j] = unalignedLoadLittleEndian<T>(sample_pos);

            Estimation best_estimation = {{0, 0}, 0};
            size_t best_size = std::numeric_limits<size_t>::max();

            for (UInt8 exponent = 0; exponent < ALPFloatTraits<T>::EXPONENT_COUNT; ++exponent)
            {
                for (UInt8 fraction = 0; fraction <= exponent; ++fraction)
                {
                    const Estimation estimation{{exponent, fraction}, 1};
                    const size_t estimated_size = estimateEncodedSize(sample, sample_float_count, estimation.params);

                    if (estimated_size < best_size)
                    {
                        best_size = estimated_size;
                        best_estimation = estimation;
                    }
                }
            }

            const UInt16 key = (static_cast<UInt16>(best_estimation.params.exponent) << 8) | static_cast<UInt16>(best_estimation.params.fraction);
            auto it = estimations_map.find(key);
            if (it != estimations_map.end())
                ++(it->second.occurred_times);
            else
                estimations_map[key] = best_estimation;
        }

        // Sort estimations by occurred times desc, exponent asc, fraction asc
        std::array<Estimation, ALP_PARAMS_ESTIMATION_SAMPLES> estimations;
        UInt32 estimation_count = 0;
        for (const auto & [_, estimation] : estimations_map)
            estimations[estimation_count++] = estimation;
        std::sort(estimations.begin(), estimations.begin() + estimation_count,
            [](const Estimation & a, const Estimation & b)
            {
                if (a.occurred_times == b.occurred_times)
                {
                    if (a.params.exponent == b.params.exponent)
                        return a.params.fraction < b.params.fraction;
                    return a.params.exponent < b.params.exponent;
                }
                return a.occurred_times > b.occurred_times;
            });

        // Keep top ALP_PARAMS_ESTIMATION_CANDIDATES candidates
        param_candidates.clear();
        estimation_count = std::min(estimation_count, ALP_PARAMS_ESTIMATION_CANDIDATES);
        for (UInt32 i = 0; i < estimation_count; ++i)
            param_candidates.push_back(estimations[i].params);
    }

    UInt32 estimateEncodedSize(const T * const source, const UInt32 float_count, const EncodingParams & params)
    {
        Int64 min = std::numeric_limits<Int64>::max();
        Int64 max = std::numeric_limits<Int64>::min();
        UInt32 exception_count = 0;

        for (UInt32 i = 0; i < float_count; ++i)
        {
            const T value = source[i];
            const Int64 value_enc = ALPFloatUtils<T>::encodeValue(value, params.exponent, params.fraction);
            const T value_dec = ALPFloatUtils<T>::decodeValue(value_enc, params.exponent, params.fraction);

            if (likely(value == value_dec))
            {
                min = std::min(value_enc, min);
                max = std::max(value_enc, max);
            }
            else
                ++exception_count;
        }

        const UInt8 bit_width = calculateBitWidth(min, max);
        const UInt32 total_size = ALP_BLOCK_HEADER_SIZE + float_count * bit_width / 8 + exception_count * (sizeof(UInt16) + sizeof(T));
        return total_size;
    }

    static UInt8 calculateBitWidth(const Int64 min_value, const Int64 max_value)
    {
        if (unlikely(min_value > max_value))
            return sizeof(Int64) * 8; // Edge case when no values are encoded or overflow happened.

        const UInt64 diff = static_cast<UInt64>(max_value - min_value);
        if (unlikely(diff == 0))
            return 0; // Edge case when all values are the same.

        const auto bit_width = sizeof(Int64) * 8 - getLeadingZeroBitsUnsafe<UInt64>(diff);
        return static_cast<UInt8>(bit_width);
    }
};

template <FLOAT T>
class ALPCodecDecoder
{
public:
    explicit ALPCodecDecoder() = default;

    void decode(const char * source, UInt32 source_size, char * dest, UInt32 uncompressed_size, UInt16 block_float_count)
    {
        if (uncompressed_size % sizeof(T) != 0)
            throw Exception(ErrorCodes::CANNOT_DECOMPRESS, "Cannot decompress ALP-encoded data. Invalid uncompressed size");

        if (block_float_count != ALP_MAX_BLOCK_FLOAT_COUNT)
            throw Exception(ErrorCodes::CANNOT_DECOMPRESS, "Cannot decompress ALP-encoded data. Supported block float count is {}", ALP_MAX_BLOCK_FLOAT_COUNT);

        const char * source_end = source + source_size;
        const char * dest_end = dest + uncompressed_size;

        while (source < source_end)
        {
            const UInt16 current_block_float_count = std::min<UInt16>(block_float_count, (dest_end - dest) / sizeof(T));
            decodeBlock(source, source_end, dest, dest_end, current_block_float_count);
        }

        assert(source == source_end);
        assert(dest == dest_end);
    }

private:
    alignas(64) Int64 block_encoded[ALP_MAX_BLOCK_FLOAT_COUNT];
    alignas(64) UInt64 block_bitpacked[ALP_MAX_BLOCK_FLOAT_COUNT];

    void decodeBlock(const char * & source, const char * source_end, char * & dest, const char * dest_end, const UInt16 float_count)
    {
        if (source + ALP_UNENCODED_BLOCK_HEADER_SIZE > source_end)
            throw Exception(ErrorCodes::CANNOT_DECOMPRESS, "Cannot decompress ALP-encoded data. Incomplete block header");

        const UInt8 exponent = static_cast<UInt8>(*source++);
        if (exponent == ALP_UNENCODED_BLOCK_EXPONENT)
        {
            processUnencodedBlock(source, source_end, dest, dest_end, float_count);
            return;
        }

        if (source + (ALP_BLOCK_HEADER_SIZE - ALP_UNENCODED_BLOCK_HEADER_SIZE) > source_end)
            throw Exception(ErrorCodes::CANNOT_DECOMPRESS, "Cannot decompress ALP-encoded data. Incomplete block header");

        const UInt8 fraction = static_cast<UInt8>(*source++);

        UInt16 exception_count = unalignedLoadLittleEndian<UInt16>(source);
        source += sizeof(UInt16);

        const UInt8 bit_width = static_cast<UInt8>(*source++);

        const Int64 frame_of_reference = unalignedLoadLittleEndian<Int64>(source);
        source += sizeof(Int64);

        UInt32 bitpacked_bytes_size = ALP::FFOR::calculateBitpackedSize(bit_width);
        memcpy(block_bitpacked, source, bitpacked_bytes_size);
        source += bitpacked_bytes_size;

        const UInt64 * unffor_in = block_bitpacked;
        auto *unffor_out =  reinterpret_cast<UInt64 *>(block_encoded);
        const auto *unffor_base_p = reinterpret_cast<const UInt64 *>(&frame_of_reference);
        ALP::FFOR::unffor(unffor_in, unffor_out, bit_width, unffor_base_p);

        char * dest_start = dest;
        for (UInt32 i = 0; i < float_count; ++i)
        {
            const T decoded_value = ALPFloatUtils<T>::decodeValue(block_encoded[i], exponent, fraction);

            unalignedStoreLittleEndian<T>(dest, decoded_value);
            dest += sizeof(T);
        }

        // Copy exceptions
        while (exception_count > 0)
        {
            const UInt16 exception_index = unalignedLoadLittleEndian<UInt16>(source);
            const T exception_value = unalignedLoadLittleEndian<T>(source + sizeof(UInt16));
            source += sizeof(UInt16) + sizeof(T);

            const UInt32 dest_offset = exception_index * sizeof(T);

            unalignedStoreLittleEndian<T>(dest_start + dest_offset, exception_value);

            --exception_count;
        }
    }

    static void processUnencodedBlock(const char * & source, const char * source_end, char * & dest, const char * dest_end, const UInt16 float_count)
    {
        const size_t block_size = float_count * sizeof(T);
        if (source + block_size > source_end || dest + block_size > dest_end)
            throw Exception(ErrorCodes::CANNOT_DECOMPRESS, "Cannot decompress ALP-encoded data. Incomplete uncompressed block");

        memcpy(dest, source, block_size);
        source += block_size;
        dest += block_size;
    }
};

}

CompressionCodecALP::CompressionCodecALP(UInt8 float_width_)
    : float_width(float_width_)
{
    setCodecDescription("ALP", {});
}

uint8_t CompressionCodecALP::getMethodByte() const
{
    return static_cast<uint8_t>(CompressionMethodByte::ALP);
}

void CompressionCodecALP::updateHash(SipHash & hash) const
{
    getCodecDesc()->updateTreeHash(hash, /* ignore_aliases */ true);
}

String CompressionCodecALP::getDescription() const
{
    return "Adaptive Lossless floating-Point; suitable for time series data.";
}

UInt32 CompressionCodecALP::getMaxCompressedDataSize(UInt32 uncompressed_size) const
{
    // Maximum possible encoding size = uncompressed data + codec header + number of blocks * block header
    const UInt32 num_blocks = uncompressed_size / float_width / ALP_MAX_BLOCK_FLOAT_COUNT + 1;
    return uncompressed_size + ALP_CODEC_HEADER_SIZE + num_blocks * ALP_BLOCK_HEADER_SIZE;
}

UInt32 CompressionCodecALP::doCompressData(const char * source, UInt32 source_size, char * dest) const
{
    // Write ALP header
    *dest++ = ALP_CODEC_VERSION; // meta_byte: version = 1, variant = 0
    *dest++ = float_width;
    unalignedStoreLittleEndian<UInt16>(dest, ALP_MAX_BLOCK_FLOAT_COUNT);
    dest += sizeof(UInt16);

    UInt32 dest_size = 0;

    switch (float_width)
    {
    case sizeof(Float32):
        dest_size = ALPCodecEncoder<Float32>().encode(source, source_size, dest);
        break;
    case sizeof(Float64):
        dest_size = ALPCodecEncoder<Float64>().encode(source, source_size, dest);
        break;
    default:
        throw Exception(ErrorCodes::CANNOT_COMPRESS,
            "Cannot compress with codec ALP. Unsupported float width {}",
            static_cast<size_t>(float_width));
    }

    return dest_size + ALP_CODEC_HEADER_SIZE;
}

void CompressionCodecALP::doDecompressData(const char * source, UInt32 source_size, char * dest, UInt32 uncompressed_size) const
{
    if (source_size < ALP_CODEC_HEADER_SIZE)
        throw Exception(ErrorCodes::CANNOT_DECOMPRESS, "Cannot decompress ALP-encoded data. File has wrong header");

    // Read ALP header
    const UInt8 meta_byte = static_cast<UInt8>(*source++);
    const UInt8 codec_version = meta_byte & 0x0F;
    const UInt8 codec_variant = meta_byte >> 4 & 0x01;

    if (codec_version != ALP_CODEC_VERSION)
        throw Exception(ErrorCodes::CANNOT_DECOMPRESS,
            "Cannot decompress ALP-encoded data. Unsupported codec version {}",
            static_cast<UInt32>(codec_version));

    if (codec_variant != 0)
        throw Exception(ErrorCodes::CANNOT_DECOMPRESS,
            "Cannot decompress ALP-encoded data. Unsupported codec variant {}",
            static_cast<UInt32>(codec_variant));

    const UInt8 data_float_width = static_cast<UInt8>(*source++);

    const UInt16 block_float_count = unalignedLoadLittleEndian<UInt16>(source);
    source += sizeof(UInt16);

    source_size -= ALP_CODEC_HEADER_SIZE;

    switch (data_float_width)
    {
    case sizeof(Float32):
        ALPCodecDecoder<Float32>().decode(source, source_size, dest, uncompressed_size, block_float_count);
        break;
    case sizeof(Float64):
        ALPCodecDecoder<Float64>().decode(source, source_size, dest, uncompressed_size, block_float_count);
        break;
    default:
        throw Exception(ErrorCodes::CANNOT_DECOMPRESS,
            "Cannot decompress ALP-encoded data. Unsupported float width {}",
            static_cast<size_t>(data_float_width));
    }
}

void registerCodecALP(CompressionCodecFactory & factory)
{
    auto method_code = static_cast<UInt8>(CompressionMethodByte::ALP);
    auto codec_builder = [&](const ASTPtr & arguments, const IDataType * column_type) -> CompressionCodecPtr
    {
        if (arguments && !arguments->children.empty())
            throw Exception(ErrorCodes::ILLEGAL_SYNTAX_FOR_CODEC_TYPE,
                "ALP codec must have 0 parameters, given {}",
                arguments->children.size());

        UInt8 float_width = sizeof(Float64);
        if (column_type)
        {
            if (!WhichDataType(column_type).isNativeFloat())
                throw Exception(ErrorCodes::BAD_ARGUMENTS,
                    "Codec ALP is not applicable for {} because the data type is not Float*",
                    column_type->getName());

            float_width = column_type->getSizeOfValueInMemory();
        }
        return std::make_shared<CompressionCodecALP>(float_width);
    };
    factory.registerCompressionCodecWithType("ALP", method_code, codec_builder);
}
}
