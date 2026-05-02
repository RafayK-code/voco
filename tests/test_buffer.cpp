#include <doctest/doctest.h>
#include "test_context.h"
#include <vector>
#include <numeric>

using namespace voco;

static constexpr BufferUsage kCopyUsage = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;

TEST_CASE("buffer creation")
{
    auto& dev = g_ctx->device();

    SUBCASE("device buffer")
    {
        Buffer buf = dev.createBuffer(BufferUsage::Storage, 256);
        CHECK(buf.valid());
        CHECK(buf.size() == 256);
        CHECK(buf.memoryType() == MemoryType::Device);
    }

    SUBCASE("host buffer")
    {
        Buffer buf = dev.createBuffer(BufferUsage::Storage, 256, MemoryType::Host);
        CHECK(buf.valid());
        CHECK(buf.isHostVisible());
    }

    SUBCASE("uniform buffer")
    {
        Buffer buf = dev.createBuffer(BufferUsage::Uniform, 64, MemoryType::Host);
        CHECK(buf.valid());
    }
}

TEST_CASE("copyToDevice / copyToHost roundtrip - device buffer (staged)")
{
    auto& dev = g_ctx->device();

    std::vector<uint32_t> src(64);
    std::iota(src.begin(), src.end(), 0u);
    const VkDeviceSize byteSize = src.size() * sizeof(uint32_t);

    Buffer buf = dev.createBuffer(kCopyUsage, byteSize);
    dev.copyToDevice(src.data(), buf, 0, byteSize);

    std::vector<uint32_t> dst(64, 0u);
    dev.copyToHost(buf, dst.data(), 0, byteSize);

    CHECK(src == dst);
}

TEST_CASE("copyToDevice / copyToHost roundtrip - host buffer (mapped)")
{
    auto& dev = g_ctx->device();

    std::vector<float> src(128);
    std::iota(src.begin(), src.end(), 0.0f);
    const VkDeviceSize byteSize = src.size() * sizeof(float);

    Buffer buf = dev.createBuffer(BufferUsage::Storage, byteSize, MemoryType::Host);
    dev.copyToDevice(src.data(), buf, 0, byteSize);

    std::vector<float> dst(128, 0.0f);
    dev.copyToHost(buf, dst.data(), 0, byteSize);

    CHECK(src == dst);
}

TEST_CASE("copyToDevice with offset")
{
    auto& dev = g_ctx->device();

    constexpr VkDeviceSize bufSize = 256;
    Buffer buf = dev.createBuffer(kCopyUsage, bufSize);

    std::vector<uint8_t> zeros(bufSize, 0);
    dev.copyToDevice(zeros.data(), buf, 0, bufSize);

    uint32_t value = 0xDEADBEEF;
    dev.copyToDevice(&value, buf, 128, sizeof(uint32_t));

    uint32_t readback = 0;
    dev.copyToHost(buf, &readback, 128, sizeof(uint32_t));
    CHECK(readback == value);

    uint32_t before = 0xFFFFFFFF;
    dev.copyToHost(buf, &before, 124, sizeof(uint32_t));
    CHECK(before == 0u);
}

TEST_CASE("async copy roundtrip")
{
    auto& dev = g_ctx->device();

    std::vector<int32_t> src(256);
    std::iota(src.begin(), src.end(), -128);
    const VkDeviceSize byteSize = src.size() * sizeof(int32_t);

    Buffer buf = dev.createBuffer(kCopyUsage, byteSize);
    dev.copyToDeviceAsync(src.data(), buf, 0, byteSize).wait();

    std::vector<int32_t> dst(256, 0);
    dev.copyToHostAsync(buf, dst.data(), 0, byteSize).wait();

    CHECK(src == dst);
}
