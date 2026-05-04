#include <doctest/doctest.h>
#include "test_context.h"
#include "test_pipelines.h"
#include <vector>

using namespace voco;

static constexpr BufferUsage kStorageCopy = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;

TEST_CASE("cross-submission RAW")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> zeros(count, 0.0f);
    Buffer buf    = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(zeros.data(), buf, 0, byteSize);

    // submission 1: write 1.0 into buf
    CommandList cmd1 = dev.createCommandList();
    cmd1.bindPipeline(inplaceAddPipeline());
    cmd1.bindBuffer(0, 0, buf, Access::ReadWrite);
    cmd1.dispatch(1, 1, 1);
    dev.submit(cmd1);

    // submission 2: read buf, write outBuf — voco inserts timeline semaphore wait
    struct PC { float scale; } pc{ 1.0f };
    CommandList cmd2 = dev.createCommandList();
    cmd2.bindPipeline(scalePipeline());
    cmd2.bindBuffer(0, 0, buf, Access::Read);
    cmd2.bindBuffer(0, 1, outBuf, Access::Write);
    cmd2.setPushConstants(pc);
    cmd2.dispatch(1, 1, 1);
    dev.submit(cmd2);

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(outBuf, result.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 1.0f);
}

TEST_CASE("copyToHost waits for in-flight dispatch")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> zeros(count, 0.0f);
    Buffer buf = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(zeros.data(), buf, 0, byteSize);

    // submit dispatch without waiting — buf is now in-flight
    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(inplaceAddPipeline());
    cmd.bindBuffer(0, 0, buf, Access::ReadWrite);
    cmd.dispatch(1, 1, 1);
    dev.submit(cmd);

    // copyToHost detects buf.m_lastSubmissionID > lastFinished and waits
    std::vector<float> result(count, 0.0f);
    dev.copyToHost(buf, result.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 1.0f);
}
