#include <doctest/doctest.h>
#include "test_context.h"
#include <optional>
#include <vector>
#include <numeric>

using namespace voco;

static constexpr BufferUsage kStorageCopy = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;

#define SHADER(name) VOCO_TEST_SHADER_DIR "/" name

static std::optional<ComputePipeline> s_scalePipeline;
static std::optional<ComputePipeline> s_inplaceAddPipeline;

static ComputePipeline& scalePipeline()
{
    if (!s_scalePipeline)
        s_scalePipeline = g_ctx->device().createComputePipeline(SHADER("test_scale.glsl"), ShaderSourceType::GLSL);
    return *s_scalePipeline;
}

static ComputePipeline& inplaceAddPipeline()
{
    if (!s_inplaceAddPipeline)
        s_inplaceAddPipeline = g_ctx->device().createComputePipeline(SHADER("test_inplace_add.glsl"), ShaderSourceType::GLSL);
    return *s_inplaceAddPipeline;
}

void cleanupTestPipelines()
{
    s_scalePipeline.reset();
    s_inplaceAddPipeline.reset();
}

TEST_CASE("pipeline creation")
{
    CHECK(scalePipeline().valid());
    CHECK(inplaceAddPipeline().valid());
}

TEST_CASE("basic dispatch")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count);
    std::iota(src.begin(), src.end(), 1.0f);

    Buffer inBuf  = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(src.data(), inBuf, 0, byteSize);

    struct PC { float scale; } pc{ 2.0f };

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
        CHECK(result[i] == src[i] * 2.0f);
}

TEST_CASE("push constants affect output")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count, 1.0f);

    Buffer inBuf  = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(src.data(), inBuf, 0, byteSize);

    struct PC { float scale; };

    for (float scale : { 0.5f, 3.0f, -1.0f })
    {
        CommandList cmd = dev.createCommandList();
        cmd.bindPipeline(scalePipeline());
        cmd.bindBuffer(0, 0, inBuf, Access::Read);
        cmd.bindBuffer(0, 1, outBuf, Access::Write);
        cmd.setPushConstants(PC{ scale });
        cmd.dispatch(1, 1, 1);
        dev.submit(cmd);

        std::vector<float> result(count, 0.0f);
        dev.copyToHost(outBuf, result.data(), 0, byteSize);

        for (uint32_t i = 0; i < count; ++i)
            CHECK(result[i] == scale);
    }
}

TEST_CASE("multi-dispatch barrier")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> zeros(count, 0.0f);
    Buffer buf = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(zeros.data(), buf, 0, byteSize);

    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(inplaceAddPipeline());
    cmd.bindBuffer(0, 0, buf, Access::ReadWrite);
    cmd.dispatch(1, 1, 1);
    cmd.bindBuffer(0, 0, buf, Access::ReadWrite);
    cmd.dispatch(1, 1, 1);
    dev.submit(cmd);

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(buf, result.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 2.0f);
}
