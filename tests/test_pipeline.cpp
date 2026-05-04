#include <doctest/doctest.h>
#include "test_context.h"
#include "test_pipelines.h"
#include <optional>
#include <vector>
#include <numeric>

using namespace voco;

static constexpr BufferUsage kStorageCopy = BufferUsage::Storage | BufferUsage::TransferSrc | BufferUsage::TransferDst;

#define SHADER(name) VOCO_TEST_SHADER_DIR "/" name

static std::optional<ComputePipeline> s_scalePipeline;
static std::optional<ComputePipeline> s_inplaceAddPipeline;
static std::optional<ComputePipeline> s_uniformPipeline;
static std::optional<ComputePipeline> s_multisetPipeline;

ComputePipeline& scalePipeline()
{
    if (!s_scalePipeline)
        s_scalePipeline = g_ctx->device().createComputePipeline(SHADER("test_scale.glsl"), ShaderSourceType::GLSL);
    return *s_scalePipeline;
}

ComputePipeline& inplaceAddPipeline()
{
    if (!s_inplaceAddPipeline)
        s_inplaceAddPipeline = g_ctx->device().createComputePipeline(SHADER("test_inplace_add.glsl"), ShaderSourceType::GLSL);
    return *s_inplaceAddPipeline;
}

ComputePipeline& uniformPipeline()
{
    if (!s_uniformPipeline)
        s_uniformPipeline = g_ctx->device().createComputePipeline(SHADER("test_uniform.glsl"), ShaderSourceType::GLSL);
    return *s_uniformPipeline;
}

ComputePipeline& multisetPipeline()
{
    if (!s_multisetPipeline)
        s_multisetPipeline = g_ctx->device().createComputePipeline(SHADER("test_multiset.glsl"), ShaderSourceType::GLSL);
    return *s_multisetPipeline;
}

void cleanupTestPipelines()
{
    s_scalePipeline.reset();
    s_inplaceAddPipeline.reset();
    s_uniformPipeline.reset();
    s_multisetPipeline.reset();
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

TEST_CASE("uniform buffer")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src(count, 3.0f);

    Buffer inBuf   = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf  = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(src.data(), inBuf, 0, byteSize);

    struct Params { float scale; float bias; } params{ 2.0f, 1.0f };
    Buffer ubo = dev.createBuffer(BufferUsage::Uniform, sizeof(Params), MemoryType::Host);
    dev.copyToDevice(&params, ubo, 0, sizeof(Params));

    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(uniformPipeline());
    cmd.bindBuffer(0, 0, inBuf, Access::Read);
    cmd.bindBuffer(0, 1, outBuf, Access::Write);
    cmd.bindBuffer(0, 2, ubo, Access::Read);
    cmd.dispatch(1, 1, 1);
    dev.submit(cmd);

    std::vector<float> result(count, 0.0f);
    dev.copyToHost(outBuf, result.data(), 0, byteSize);

    // 3.0 * 2.0 + 1.0 = 7.0
    for (uint32_t i = 0; i < count; ++i)
        CHECK(result[i] == 7.0f);
}

TEST_CASE("multi-set bindings with per-dispatch rebind")
{
    auto& dev = g_ctx->device();

    constexpr uint32_t count = 64;
    constexpr VkDeviceSize byteSize = count * sizeof(float);

    std::vector<float> src1(count, 2.0f);
    std::vector<float> src2(count, 6.0f);

    Buffer inBuf1  = dev.createBuffer(kStorageCopy, byteSize);
    Buffer inBuf2  = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf1 = dev.createBuffer(kStorageCopy, byteSize);
    Buffer outBuf2 = dev.createBuffer(kStorageCopy, byteSize);
    dev.copyToDevice(src1.data(), inBuf1, 0, byteSize);
    dev.copyToDevice(src2.data(), inBuf2, 0, byteSize);

    struct Params { float scale; float bias; } params{ 2.0f, 1.0f };
    Buffer ubo = dev.createBuffer(BufferUsage::Uniform, sizeof(Params), MemoryType::Host);
    dev.copyToDevice(&params, ubo, 0, sizeof(Params));

    CommandList cmd = dev.createCommandList();
    cmd.bindPipeline(multisetPipeline());

    // dispatch 1: bind both sets
    cmd.bindBuffer(0, 0, inBuf1,  Access::Read);
    cmd.bindBuffer(0, 1, outBuf1, Access::Write);
    cmd.bindBuffer(1, 0, ubo,     Access::Read);
    cmd.dispatch(1, 1, 1);

    // dispatch 2: only rebind set 0 — set 1 (ubo) stays bound
    cmd.bindBuffer(0, 0, inBuf2,  Access::Read);
    cmd.bindBuffer(0, 1, outBuf2, Access::Write);
    cmd.dispatch(1, 1, 1);

    dev.submit(cmd);

    std::vector<float> result1(count, 0.0f);
    std::vector<float> result2(count, 0.0f);
    dev.copyToHost(outBuf1, result1.data(), 0, byteSize);
    dev.copyToHost(outBuf2, result2.data(), 0, byteSize);

    for (uint32_t i = 0; i < count; ++i)
    {
        CHECK(result1[i] == 5.0f);   // 2.0 * 2.0 + 1.0
        CHECK(result2[i] == 13.0f);  // 6.0 * 2.0 + 1.0
    }
}
