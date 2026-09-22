#include <doctest/doctest.h>
#include "test_context.h"
#include "test_pipelines.h"
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

TEST_CASE("buffer destroyed while submission in-flight")
{
    // Buffer goes out of scope right after submit, while the GPU may still be
    // executing. The retirement queue must keep the VkBuffer alive until the
    // submission completes, or the GPU reads freed memory.
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    Buffer outBuf = dev.createBuffer(kCopyUsage, byteSize);
    std::vector<float> zeros(count, 0.0f);
    dev.copyToDevice(zeros.data(), outBuf, 0, byteSize);

    struct PC { float scale; } pc{ 2.0f };

    {
        std::vector<float> src(count, 3.0f);
        Buffer shortLivedIn = dev.createBuffer(kCopyUsage, byteSize);
        dev.copyToDevice(src.data(), shortLivedIn, 0, byteSize);

        CommandList cmd = dev.createCommandList();
        cmd.bindPipeline(scalePipeline());
        cmd.bindBuffer(0, 0, shortLivedIn, Access::Read);
        cmd.bindBuffer(0, 1, outBuf, Access::Write);
        cmd.setPushConstants(pc);
        cmd.dispatch(1, 1, 1);
        dev.submit(cmd);
        // shortLivedIn destroyed here; submission still in-flight
    }

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(outBuf, result.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 6.0f);
}

TEST_CASE("buffer destroyed while cached in descriptor set")
{
    // A dying buffer no longer needs any descriptor-heap callback at all (see
    // Buffer::destroy()) -- this just confirms rapid create/dispatch/destroy cycles
    // stay correct across runs.
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);
    struct PC { float scale; } pc{ 2.0f };

    auto runOnce = [&](float inputValue, float expected) {
        std::vector<float> src(count, inputValue);
        Buffer inBuf  = dev.createBuffer(kCopyUsage, byteSize);
        Buffer outBuf = dev.createBuffer(kCopyUsage, byteSize);
        dev.copyToDevice(src.data(), inBuf, 0, byteSize);

        CommandList cmd = dev.createCommandList();
        cmd.bindPipeline(scalePipeline());
        cmd.bindBuffer(0, 0, inBuf, Access::Read);
        cmd.bindBuffer(0, 1, outBuf, Access::Write);
        cmd.setPushConstants(pc);
        cmd.dispatch(1, 1, 1);
        dev.submit(cmd);

        std::vector<float> result(count, 0.0f);
        dev.copyToHost(outBuf, result.data(), 0, byteSize);
        for (uint32_t i = 0; i < count; ++i)
            CHECK(result[i] == expected);
        // inBuf and outBuf destroyed here — must not affect subsequent runs.
    };

    runOnce(1.0f, 2.0f);
    runOnce(4.0f, 8.0f);
    runOnce(9.0f, 18.0f);
}

TEST_CASE("buffer destroyed with sibling — sibling remains usable")
{
    // doomed and survivor are bound together in one dispatch; when doomed dies,
    // survivor must remain fully usable afterward.
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);
    struct PC { float scale; } pc{ 2.0f };

    Buffer survivor = dev.createBuffer(kCopyUsage, byteSize);

    {
        std::vector<float> src(count, 7.0f);
        Buffer doomed = dev.createBuffer(kCopyUsage, byteSize);
        dev.copyToDevice(src.data(), doomed, 0, byteSize);

        CommandList cmd = dev.createCommandList();
        cmd.bindPipeline(scalePipeline());
        cmd.bindBuffer(0, 0, doomed, Access::Read);
        cmd.bindBuffer(0, 1, survivor, Access::Write);
        cmd.setPushConstants(pc);
        cmd.dispatch(1, 1, 1);
        dev.submit(cmd);

        std::vector<float> verify(count, 0.0f);
        dev.copyToHost(survivor, verify.data(), 0, byteSize);
        for (uint32_t i = 0; i < count; ++i)
            CHECK(verify[i] == 14.0f);
        // doomed destroyed here — survivor's prior dispatch already retired
    }

    std::vector<float> src(count, 11.0f);
    Buffer newIn = dev.createBuffer(kCopyUsage, byteSize);
    dev.copyToDevice(src.data(), newIn, 0, byteSize);

    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(scalePipeline());
    cmd.bindBuffer(0, 0, newIn, Access::Read);
    cmd.bindBuffer(0, 1, survivor, Access::Write);
    cmd.setPushConstants(pc);
    cmd.dispatch(1, 1, 1);
    dev.submit(cmd);

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(survivor, result.data(), 0, byteSize);
    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 22.0f);
}
