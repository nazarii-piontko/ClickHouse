#include <Compression/FFOR.h>

#include <random>

#include <gtest/gtest.h>

using namespace DB;

namespace
{
class FFORTest : public ::testing::TestWithParam<UInt16> {};

TEST_P(FFORTest, FFORTestForBits)
{
    const auto bits = GetParam();
    ASSERT_LE(bits, 64);

    alignas(64) UInt64 in[Compression::FFOR::DEFAULT_VALUES];
    alignas(64) UInt64 coded[Compression::FFOR::DEFAULT_VALUES];
    alignas(64) UInt64 decoded[Compression::FFOR::DEFAULT_VALUES];
    constexpr UInt64 base = 13;

    // Generate test data
    const UInt64 max_value = (1ULL << std::min<UInt8>(bits, 63u)) - 1ULL;
    std::default_random_engine rng(0xC0FFEEULL);
    std::uniform_int_distribution<UInt64> dist(0, max_value);
    for (UInt16 i = 0; i < Compression::FFOR::DEFAULT_VALUES; ++i)
        in[i] = base + dist(rng);

    // Encode
    Compression::FFOR::bitPack(in, coded, bits, base);

    // Set unused bits to zero to ensure decoder does not rely on them
    UInt16 used_bytes = Compression::FFOR::calculateBitpackedBytes(bits);
    std::memset(reinterpret_cast<char*>(coded) + used_bytes, 0, sizeof(coded) - used_bytes);

    // Decode
    Compression::FFOR::bitUnpack(coded, decoded, bits, base);

    // Verify
    for (UInt16 i = 0; i < Compression::FFOR::DEFAULT_VALUES; ++i)
        ASSERT_EQ(decoded[i], in[i]) << "bits=" << static_cast<UInt32>(bits) << " index=" << i << " in=" << in[i] << " decoded=" << decoded[i];
}

INSTANTIATE_TEST_SUITE_P(FFORTest,
    FFORTest,
    ::testing::Range<UInt16>(0, 65)
);
}
