// Exercises the concurrent-use contract from CLAUDE.md. These are best-effort
// guardrails, not race-detectors — a pass doesn't prove race-freedom. Run
// under TSan (clang-cl or WSL) to actually catch data races; on MSVC they
// only catch crashes, deadlocks, and validation-layer errors.

#include <doctest/doctest.h>
#include "test_context.h"
#include "test_pipelines.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

using namespace voco;

static constexpr BufferUsage kStorageCopy = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;

#define SHADER(name) VOCO_TEST_SHADER_DIR "/" name

TEST_CASE("threading: concurrent buffer create/destroy")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t threadCount = 8;
    constexpr uint32_t iterations = 200;

    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([&]() {
            for (uint32_t i = 0; i < iterations; ++i)
            {
                Buffer b = dev.createBuffer(BufferUsage::Storage, 256);
                if (!b.valid()) { failed = true; return; }
                // ~Buffer() runs at end of scope — exercises Buffer::destroy() concurrently
            }
        });
    }

    for (auto& th : threads) th.join();
    CHECK_FALSE(failed.load());
}

TEST_CASE("threading: concurrent pipeline creation")
{
    auto& dev = g_ctx->device();

    // Pre-warm shaderc / global pipeline statics on the main thread.
    scalePipeline();

    constexpr uint32_t threadCount = 4;
    constexpr uint32_t pipelinesPerThread = 4;

    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;

    // All threads compile the same shader independently (no shared layout cache under
    // the heap backend) — this exercises parallel pipeline creation.
    for (uint32_t t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([&]() {
            for (uint32_t i = 0; i < pipelinesPerThread; ++i)
            {
                ComputePipeline p = dev.createComputePipeline(SHADER("test_scale.glsl"), ShaderSourceType::GLSL);
                if (!p.valid()) { failed = true; return; }
            }
        });
    }

    for (auto& th : threads) th.join();
    CHECK_FALSE(failed.load());
}

TEST_CASE("threading: concurrent record, serial submit")
{
    auto& dev = g_ctx->device();
    scalePipeline();  // pre-warm

    constexpr uint32_t threadCount = 8;
    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    // Each thread owns its own buffer pair — a single Buffer is not thread-safe.
    std::vector<Buffer> inBufs;
    std::vector<Buffer> outBufs;
    inBufs.reserve(threadCount);
    outBufs.reserve(threadCount);
    for (uint32_t t = 0; t < threadCount; ++t)
    {
        inBufs.push_back(dev.createBuffer(kStorageCopy, byteSize));
        outBufs.push_back(dev.createBuffer(kStorageCopy, byteSize));
        std::vector<float> src(count, static_cast<float>(t + 1));
        dev.copyToDevice(src.data(), inBufs.back(), 0, byteSize);
    }

    struct PC { float scale; } pc{ 2.0f };

    std::mutex submitMutex;
    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([&, t]() {
            CommandList cmd = dev.createCommandList();
            cmd.bindPipeline(scalePipeline());
            cmd.bindBuffer(0, 0, inBufs[t], Access::Read);
            cmd.bindBuffer(0, 1, outBufs[t], Access::Write);
            cmd.setPushConstants(pc);
            cmd.dispatch(1, 1, 1);

            std::lock_guard<std::mutex> lock(submitMutex);
            dev.submit(cmd);
        });
    }

    for (auto& th : threads) th.join();

    for (uint32_t t = 0; t < threadCount; ++t)
    {
        std::vector<float> result(count, 0.0f);
        dev.copyToHost(outBufs[t], result.data(), 0, byteSize);
        const float expected = static_cast<float>(t + 1) * 2.0f;
        for (uint32_t i = 0; i < count; ++i)
            CHECK(result[i] == expected);
    }
}

TEST_CASE("threading: mixed churn and recording")
{
    // Worst-case stress: some threads churn short-lived buffer pairs through
    // dispatches (exercising heap-chunk acquisition while a submission is still
    // in-flight), others reuse long-lived buffers. Verifies no crash / no
    // validation errors under concurrent cache reads + writes.
    auto& dev = g_ctx->device();
    scalePipeline();

    constexpr uint32_t churnThreads = 4;
    constexpr uint32_t workerThreads = 4;
    constexpr uint32_t iterations = 100;
    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::mutex submitMutex;
    std::atomic<bool> failed{false};

    std::vector<Buffer> workerInBufs;
    std::vector<Buffer> workerOutBufs;
    for (uint32_t t = 0; t < workerThreads; ++t)
    {
        workerInBufs.push_back(dev.createBuffer(kStorageCopy, byteSize));
        workerOutBufs.push_back(dev.createBuffer(kStorageCopy, byteSize));
    }

    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < churnThreads; ++t)
    {
        threads.emplace_back([&]() {
            struct PC { float scale; } pc{ 1.0f };
            for (uint32_t i = 0; i < iterations; ++i)
            {
                Buffer inBuf = dev.createBuffer(kStorageCopy, byteSize);
                Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);

                CommandList cmd = dev.createCommandList();
                cmd.bindPipeline(scalePipeline());
                cmd.bindBuffer(0, 0, inBuf, Access::Read);
                cmd.bindBuffer(0, 1, outBuf, Access::Write);
                cmd.setPushConstants(pc);
                cmd.dispatch(1, 1, 1);

                {
                    std::lock_guard<std::mutex> lock(submitMutex);
                    dev.submit(cmd);
                }
                // buffers destroyed here, possibly while submission is still in-flight —
                // the retirement queue holds the VkBuffer alive until the GPU finishes.
            }
        });
    }

    for (uint32_t t = 0; t < workerThreads; ++t)
    {
        threads.emplace_back([&, t]() {
            struct PC { float scale; } pc{ 1.0f };
            for (uint32_t i = 0; i < iterations; ++i)
            {
                CommandList cmd = dev.createCommandList();
                cmd.bindPipeline(scalePipeline());
                cmd.bindBuffer(0, 0, workerInBufs[t], Access::Read);
                cmd.bindBuffer(0, 1, workerOutBufs[t], Access::Write);
                cmd.setPushConstants(pc);
                cmd.dispatch(1, 1, 1);

                std::lock_guard<std::mutex> lock(submitMutex);
                dev.submit(cmd);
            }
        });
    }

    for (auto& th : threads) th.join();
    CHECK_FALSE(failed.load());
}