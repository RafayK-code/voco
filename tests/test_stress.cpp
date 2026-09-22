#include <doctest/doctest.h>
#include "test_context.h"
#include "test_pipelines.h"
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>

using namespace voco;

static constexpr BufferUsage kStorageCopy = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;

// ---- helpers ----------------------------------------------------------------

static double ms(std::chrono::high_resolution_clock::time_point a,
                 std::chrono::high_resolution_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---- tests ------------------------------------------------------------------

TEST_CASE("stress: sequential submissions")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr uint32_t iterations = 1000;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count, 1.0f);
    Buffer inBuf = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(src.data(), inBuf, 0, byteSize);

    struct PC { float scale; } pc{ 2.0f };

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i)
    {
        CommandList cmd = dev.createCommandList();
        cmd.bindPipeline(scalePipeline());
        cmd.bindBuffer(0, 0, inBuf, Access::Read);
        cmd.bindBuffer(0, 1, outBuf, Access::Write);
        cmd.setPushConstants(pc);
        cmd.dispatch(1, 1, 1);
        dev.submit(cmd);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(outBuf, result.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 2.0f);

    double total = ms(t0, t1);
    std::cout << "[sequential submissions] " << iterations << " submissions in "
              << total << " ms  (" << total / iterations << " ms/submit)\n";
}

TEST_CASE("stress: buffer churn")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr uint32_t iterations = 500;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count, 3.0f);
    struct PC { float scale; } pc{ 1.0f };

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iterations; ++i)
    {
        //std::cout << "new iter\n";
        Buffer inBuf = dev.createBuffer(kStorageCopy, byteSize);
        Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);
        dev.copyToDevice(src.data(), inBuf, 0, byteSize);

        CommandList cmd = dev.createCommandList();
        cmd.bindPipeline(scalePipeline());
        cmd.bindBuffer(0, 0, inBuf, Access::Read);
        cmd.bindBuffer(0, 1, outBuf, Access::Write);
        cmd.setPushConstants(pc);
        cmd.dispatch(1, 1, 1);
        dev.submit(cmd);
        // buffers destroyed here — exercises Buffer::destroy() + retirement queue
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total = ms(t0, t1);
    std::cout << "[buffer churn] " << iterations << " create/dispatch/destroy cycles in "
              << total << " ms  (" << total / iterations << " ms/cycle)\n";
}

TEST_CASE("stress: large data throughput")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 1 << 20; // 1M floats = 4 MB
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count);
    std::iota(src.begin(), src.end(), 1.0f);

    Buffer inBuf = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);

    auto tUpload = std::chrono::high_resolution_clock::now();
    dev.copyToDevice(src.data(), inBuf, 0, byteSize);

    struct PC { float scale; } pc{ 2.0f };

    auto tDispatch = std::chrono::high_resolution_clock::now();
    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(scalePipeline());
    cmd.bindBuffer(0, 0, inBuf, Access::Read);
    cmd.bindBuffer(0, 1, outBuf, Access::Write);
    cmd.setPushConstants(pc);
    cmd.dispatch(count / 64, 1, 1);
    dev.submit(cmd);

    std::vector<float> result(count, 0.0f);
    auto tDownload = std::chrono::high_resolution_clock::now();
    dev.copyToHost(outBuf, result.data(), 0, byteSize);
    auto tDone = std::chrono::high_resolution_clock::now();

    bool correct = true;
    for (uint32_t i = 0; i < count; ++i)
        if (result[i] != src[i] * 2.0f) { correct = false; break; }
    CHECK(correct);

    double uploadMs = ms(tUpload, tDispatch);
    double dispatchMs = ms(tDispatch, tDownload);
    double downloadMs = ms(tDownload, tDone);
    double totalMs = ms(tUpload, tDone);
    // read + write on the GPU side
    double gbps = (2.0 * byteSize) / (dispatchMs * 1e-3) / 1e9;

    std::cout << "[large data]  upload " << uploadMs << " ms  |  dispatch+wait "
              << dispatchMs << " ms  (" << gbps << " GB/s)  |  download "
              << downloadMs << " ms  |  total " << totalMs << " ms\n";
}

TEST_CASE("stress: descriptor rebinding single command list")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr uint32_t numPairs = 200;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count, 1.0f);

    std::vector<Buffer> inBufs;
    std::vector<Buffer> outBufs;
    inBufs.reserve(numPairs);
    outBufs.reserve(numPairs);

    for (uint32_t i = 0; i < numPairs; ++i)
    {
        inBufs.push_back(dev.createBuffer(kStorageCopy, byteSize));
        outBufs.push_back(dev.createBuffer(kStorageCopy, byteSize));

        std::fill(src.begin(), src.end(), static_cast<float>(i + 1));
        dev.copyToDevice(src.data(), inBufs.back(), 0, byteSize);
    }

    struct PC { float scale; } pc{ 2.0f };

    auto t0 = std::chrono::high_resolution_clock::now();

    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(scalePipeline());

    for (uint32_t i = 0; i < numPairs; ++i)
    {
        cmd.bindBuffer(0, 0, inBufs[i], Access::Read);
        cmd.bindBuffer(0, 1, outBufs[i], Access::Write);
        cmd.setPushConstants(pc);
        cmd.dispatch(1, 1, 1);
    }

    dev.submit(cmd);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Spot-check first, middle, and last.
    const uint32_t checks[] = {
        0,
        numPairs / 2,
        numPairs - 1
    };

    for (uint32_t idx : checks)
    {
        std::vector<float> result(count, 0.0f);
        dev.copyToHost(outBufs[idx], result.data(), 0, byteSize);

        const float expected = static_cast<float>(idx + 1) * 2.0f;

        for (uint32_t j = 0; j < count; ++j)
            CHECK(result[j] == doctest::Approx(expected));
    }

    double total = ms(t0, t1);
    std::cout << "[descriptor rebinding single cmd] "
        << numPairs << " unique bindings/dispatches in one command list in "
        << total << " ms  ("
        << total / numPairs << " ms/binding+dispatch)\n";
}

TEST_CASE("stress: multi-dispatch same bindings")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr uint32_t dispatches = 1000;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count, 1.0f);

    Buffer inBuf = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);

    dev.copyToDevice(src.data(), inBuf, 0, byteSize);

    struct PC { float scale; } pc{ 2.0f };

    auto t0 = std::chrono::high_resolution_clock::now();

    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(scalePipeline());
    cmd.bindBuffer(0, 0, inBuf, Access::Read);
    cmd.bindBuffer(0, 1, outBuf, Access::Write);
    cmd.setPushConstants(pc);

    for (uint32_t i = 0; i < dispatches; ++i)
        cmd.dispatch(1, 1, 1);

    dev.submit(cmd);

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(outBuf, result.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == doctest::Approx(2.0f));

    double total = ms(t0, t1);
    std::cout << "[multi-dispatch same bindings] "
        << dispatches << " dispatches in one command list in "
        << total << " ms  ("
        << total / dispatches << " ms/dispatch)\n";
}