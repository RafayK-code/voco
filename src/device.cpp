#include <voco/device.h>
#include "queue.h"
#include "retirement_queue.h"
#include "descriptor.h"
#include "buffer_registry.h"
#include "utils.h"
#include <spirv-reflect/spirv_reflect.h>
#include <fstream>
#include <sstream>
#include <unordered_set>
#ifdef VOCO_ENABLE_GLSL
#include <shaderc/shaderc.hpp>
#endif

namespace voco
{
    Device::Device(const Context& context)
        : m_ctx(context)
    {
        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = m_ctx.instance;
        allocatorInfo.physicalDevice = m_ctx.physicalDevice;
        allocatorInfo.device = m_ctx.device;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;

        vmaCreateAllocator(&allocatorInfo, &m_allocator);

        VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1000;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;

        vkCreateDescriptorPool(m_ctx.device, &poolInfo, nullptr, &m_descriptorPool);

        m_queue = std::make_unique<detail::Queue>(m_ctx.device, m_ctx.computeQueue, m_ctx.computeQueueFamilyIndex);
        m_retirementQueue = std::make_unique<detail::RetirementQueue>(*m_queue);
        m_descriptorLayoutCache = std::make_unique<detail::DescriptorLayoutCache>(m_ctx.device, m_descriptorPool, *m_retirementQueue, m_descriptorPoolMutex);
        m_pipelineLayoutCache = std::make_unique<detail::PipelineLayoutCache>(m_ctx.device, *m_descriptorLayoutCache, *m_retirementQueue);
        m_bufferRegistry = std::make_unique<detail::BufferRegistry>(*m_descriptorLayoutCache);
    }

    Device::~Device()
    {
        m_bufferRegistry.reset();
        m_pipelineLayoutCache.reset();
        m_descriptorLayoutCache.reset();
        m_retirementQueue.reset();
        m_queue.reset();

        vkDestroyDescriptorPool(m_ctx.device, m_descriptorPool, nullptr);
        vmaDestroyAllocator(m_allocator);
    }

    Buffer Device::createBuffer(BufferUsage usage, VkDeviceSize size, MemoryType memType)
    {
        auto memInfo = detail::toVmaMemoryInfo(memType);

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memInfo.usage;
        allocInfo.flags = memInfo.flags;

        VkBuffer vkBuffer;
        VmaAllocation allocation;
        VkResult res = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &vkBuffer, &allocation, nullptr);
        VK_CHECK(res);

        return Buffer(m_allocator, m_retirementQueue.get(), m_bufferRegistry.get(), vkBuffer, allocation, size, usage, memType);
    }

    CommandList Device::createCommandList()
    {
        detail::TrackedCommandBuffer cmdBuf = m_queue->acquire();

        return CommandList(m_ctx.device, std::move(cmdBuf), m_descriptorLayoutCache.get(), m_bufferRegistry.get(), m_queue->getLastFinishedID());
    }

    ComputePipeline Device::createComputePipeline(std::string_view shaderPath, ShaderSourceType sourceType)
    {
        std::vector<uint32_t> spirv;

        if (sourceType == ShaderSourceType::SPIRV)
        {
            std::ifstream file(std::string{shaderPath}, std::ios::binary | std::ios::ate);
            size_t byteSize = static_cast<size_t>(file.tellg());
            file.seekg(0);
            spirv.resize(byteSize / sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(spirv.data()), byteSize);
        }
        else if (sourceType == ShaderSourceType::GLSL)
        {
#ifdef VOCO_ENABLE_GLSL
            std::ifstream file(std::string{shaderPath});
            std::ostringstream ss;
            ss << file.rdbuf();
            std::string source = ss.str();

            shaderc::Compiler compiler;
            shaderc::CompileOptions options;
            options.SetOptimizationLevel(shaderc_optimization_level_performance);

            shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
                source.c_str(), source.size(),
                shaderc_glsl_compute_shader,
                std::string{shaderPath}.c_str(),
                "main", options);

            DEBUG_ASSERT(result.GetCompilationStatus() == shaderc_compilation_status_success,
                         "GLSL shader compilation failed");

            spirv.assign(result.cbegin(), result.cend());
#else
            DEBUG_ASSERT(false, "VOCO_ENABLE_GLSL must be enabled to compile GLSL shaders");
#endif
        }
        else if (sourceType == ShaderSourceType::HLSL)
        {
#ifdef VOCO_ENABLE_HLSL
            DEBUG_ASSERT(false, "HLSL compilation not yet implemented");
#else
            DEBUG_ASSERT(false, "VOCO_ENABLE_HLSL must be enabled to compile HLSL shaders");
#endif
        }

        SpvReflectShaderModule reflModule{};
        spvReflectCreateShaderModule(spirv.size() * sizeof(uint32_t), spirv.data(), &reflModule);

        uint32_t setCount = 0;
        spvReflectEnumerateDescriptorSets(&reflModule, &setCount, nullptr);
        std::vector<SpvReflectDescriptorSet*> reflSets(setCount);
        if (setCount > 0)
            spvReflectEnumerateDescriptorSets(&reflModule, &setCount, reflSets.data());

        uint32_t maxSet = 0;
        for (auto* s : reflSets)
            maxSet = std::max(maxSet, s->set);
        uint32_t totalSets = setCount > 0 ? maxSet + 1 : 0;

        std::vector<VkDescriptorSetLayout> setLayouts(totalSets);
        std::vector<std::vector<detail::BindingDesc>> allBindings(totalSets);

        for (auto* reflSet : reflSets)
        {
            std::vector<detail::BindingDesc> bindings;
            for (uint32_t i = 0; i < reflSet->binding_count; ++i)
            {
                auto* b = reflSet->bindings[i];
                bindings.push_back({
                    .binding = b->binding,
                    .type = static_cast<VkDescriptorType>(b->descriptor_type),
                    .stages = VK_SHADER_STAGE_COMPUTE_BIT
                });
            }
            setLayouts[reflSet->set] = m_descriptorLayoutCache->getOrCreate(bindings).layout;
            allBindings[reflSet->set] = std::move(bindings);
        }

        uint32_t pushConstSize = 0;
        if (reflModule.push_constant_block_count > 0)
            pushConstSize = reflModule.push_constant_blocks[0].size;

        spvReflectDestroyShaderModule(&reflModule);

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = pushConstSize;

        VkPipelineLayout pipelineLayout = m_pipelineLayoutCache->getOrCreate(
            allBindings, pushConstSize > 0 ? std::optional(pcRange) : std::nullopt);

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
        shaderInfo.pCode = spirv.data();

        VkShaderModule shaderModule;
        VkResult res = vkCreateShaderModule(m_ctx.device, &shaderInfo, nullptr, &shaderModule);
        VK_CHECK(res);

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;

        VkPipeline pipeline;
        res = vkCreateComputePipelines(m_ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        VK_CHECK(res);

        ComputePipeline result;
        result.m_device = m_ctx.device;
        result.m_retirementQueue = m_retirementQueue.get();
        result.m_shaderModule = shaderModule;
        result.m_descSetLayouts = std::move(setLayouts);
        result.m_pipelineLayout = pipelineLayout;
        result.m_pipeline = pipeline;
        result.m_pushConstSize = pushConstSize;
        return result;
    }

    void Device::submit(CommandList& cmd)
    {
        uint64_t lastFinished = m_queue->getLastFinishedID();
        uint64_t maxHazardID = 0;

        for (auto& bb : cmd.m_boundBuffers)
            if (bb.buffer->m_lastSubmissionID > lastFinished)
                maxHazardID = std::max(maxHazardID, bb.buffer->m_lastSubmissionID);

        if (maxHazardID > 0)
            m_queue->addWaitSemaphore(m_queue->getTimelineSemaphore(), maxHazardID);

        vkEndCommandBuffer(cmd.m_cmdBuf->cmd);
        uint64_t submissionID = m_queue->submit(std::move(*cmd.m_cmdBuf));
        cmd.m_cmdBuf.reset();

        for (auto& bb : cmd.m_boundBuffers)
            bb.buffer->m_lastSubmissionID = submissionID;

        std::unordered_set<VkPipelineLayout> usedLayouts;
        for (auto* pipeline : cmd.m_boundPipelines)
        {
            pipeline->m_lastSubmissionID = submissionID;
            VkPipelineLayout layout = pipeline->pipelineLayout();
            if (usedLayouts.insert(layout).second)
                m_pipelineLayoutCache->markSubmitted(layout, submissionID);
        }

        for (auto& used : cmd.m_usedSets)
        {
            detail::DescriptorSetKey key;
            key.bindings = used.key;
            used.cache->markSubmitted(key, submissionID);
        }
    }

    static std::pair<VkBuffer, VmaAllocation> allocStaging(VmaAllocator allocator, VkDeviceSize size, bool forRead, void** outMapped)
    {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = size;
        bufInfo.usage = forRead ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
            (forRead ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                     : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        VkBuffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo info{};

        VkResult res = vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &buffer, &allocation, &info);
        VK_CHECK(res);

        *outMapped = info.pMappedData;
        return { buffer, allocation };
    }

    static uint64_t submitCopy(detail::Queue& queue, VkBuffer src, VkBuffer dst,
                               VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        auto cb = queue.acquire();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb.cmd, &beginInfo);

        VkBufferCopy region{};
        region.srcOffset = srcOffset;
        region.dstOffset = dstOffset;
        region.size = size;
        vkCmdCopyBuffer(cb.cmd, src, dst, 1, &region);

        vkEndCommandBuffer(cb.cmd);
        return queue.submit(std::move(cb));
    }

    void Device::copyToDevice(const void* src, Buffer& dst, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        if (dst.m_lastSubmissionID > m_queue->getLastFinishedID())
            m_queue->wait(dst.m_lastSubmissionID);

        if (dst.isHostVisible())
            copyToDeviceMapped(src, dst.m_buffer, dst.m_allocation, dstOffset, size);
        else
        {
            copyToDeviceStaged(src, dst.m_buffer, dstOffset, size);
            dst.m_lastSubmissionID = m_queue->getLastSubmittedID();
        }
    }

    Future Device::copyToDeviceAsync(const void* src, Buffer& dst, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        if (dst.isHostVisible())
        {
            if (dst.m_lastSubmissionID > m_queue->getLastFinishedID())
                m_queue->wait(dst.m_lastSubmissionID);
            copyToDeviceMappedAsync(src, dst.m_buffer, dst.m_allocation, dstOffset, size);
            return Future(m_queue.get(), 0);
        }

        if (dst.m_lastSubmissionID > m_queue->getLastFinishedID())
            m_queue->addWaitSemaphore(m_queue->getTimelineSemaphore(), dst.m_lastSubmissionID);

        Future f = copyToDeviceStagedAsync(src, dst.m_buffer, dstOffset, size);
        dst.m_lastSubmissionID = m_queue->getLastSubmittedID();
        return f;
    }

    void Device::copyToHost(Buffer& src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size)
    {
        if (src.m_lastSubmissionID > m_queue->getLastFinishedID())
            m_queue->wait(src.m_lastSubmissionID);

        if (src.isHostVisible())
            copyToHostMapped(src.m_buffer, src.m_allocation, dst, srcOffset, size);
        else
            copyToHostStaged(src.m_buffer, dst, srcOffset, size);
    }

    Future Device::copyToHostAsync(Buffer& src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size)
    {
        if (src.isHostVisible())
        {
            if (src.m_lastSubmissionID > m_queue->getLastFinishedID())
                m_queue->wait(src.m_lastSubmissionID);
            copyToHostMapped(src.m_buffer, src.m_allocation, dst, srcOffset, size);
            return Future(m_queue.get(), 0);
        }

        if (src.m_lastSubmissionID > m_queue->getLastFinishedID())
            m_queue->addWaitSemaphore(m_queue->getTimelineSemaphore(), src.m_lastSubmissionID);

        Future f = copyToHostStagedAsync(src.m_buffer, dst, srcOffset, size);
        src.m_lastSubmissionID = m_queue->getLastSubmittedID();
        return f;
    }

    void Device::copyToDeviceMapped(const void* src, VkBuffer dst, VmaAllocation allocation, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        void* mapped;
        VkResult res = vmaMapMemory(m_allocator, allocation, &mapped);
        VK_CHECK(res);
        memcpy(static_cast<uint8_t*>(mapped) + dstOffset, src, size);
        vmaUnmapMemory(m_allocator, allocation);
        vmaFlushAllocation(m_allocator, allocation, dstOffset, size);
    }

    void Device::copyToHostMapped(VkBuffer src, VmaAllocation allocation, void* dst, VkDeviceSize srcOffset, VkDeviceSize size)
    {
        vmaInvalidateAllocation(m_allocator, allocation, srcOffset, size);
        void* mapped;
        VkResult res = vmaMapMemory(m_allocator, allocation, &mapped);
        VK_CHECK(res);
        memcpy(dst, static_cast<uint8_t*>(mapped) + srcOffset, size);
        vmaUnmapMemory(m_allocator, allocation);
    }

    void Device::copyToDeviceStaged(const void* src, VkBuffer dst, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        void* mapped;
        auto [stagingBuf, stagingAlloc] = allocStaging(m_allocator, size, false, &mapped);
        memcpy(mapped, src, size);
        vmaFlushAllocation(m_allocator, stagingAlloc, 0, size);

        uint64_t id = submitCopy(*m_queue, stagingBuf, dst, 0, dstOffset, size);
        m_queue->wait(id);
        vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);
    }

    void Device::copyToHostStaged(VkBuffer src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size)
    {
        void* mapped;
        auto [stagingBuf, stagingAlloc] = allocStaging(m_allocator, size, true, &mapped);

        uint64_t id = submitCopy(*m_queue, src, stagingBuf, srcOffset, 0, size);
        m_queue->wait(id);

        vmaInvalidateAllocation(m_allocator, stagingAlloc, 0, size);
        memcpy(dst, mapped, size);
        vmaDestroyBuffer(m_allocator, stagingBuf, stagingAlloc);
    }

    Future Device::copyToDeviceMappedAsync(const void* src, VkBuffer dst, VmaAllocation allocation, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        copyToDeviceMapped(src, dst, allocation, dstOffset, size);
        return Future(m_queue.get(), 0);
    }

    Future Device::copyToHostMappedAsync(VkBuffer src, VmaAllocation allocation, void* dst, VkDeviceSize srcOffset, VkDeviceSize size)
    {
        copyToHostMapped(src, allocation, dst, srcOffset, size);
        return Future(m_queue.get(), 0);
    }

    Future Device::copyToDeviceStagedAsync(const void* src, VkBuffer dst, VkDeviceSize dstOffset, VkDeviceSize size)
    {
        void* mapped;
        auto [stagingBuf, stagingAlloc] = allocStaging(m_allocator, size, false, &mapped);
        memcpy(mapped, src, size);
        vmaFlushAllocation(m_allocator, stagingAlloc, 0, size);

        uint64_t id = submitCopy(*m_queue, stagingBuf, dst, 0, dstOffset, size);

        VmaAllocator allocator = m_allocator;
        m_retirementQueue->push({ id, [allocator, stagingBuf, stagingAlloc]() {
            vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        }});

        return Future(m_queue.get(), id);
    }

    Future Device::copyToHostStagedAsync(VkBuffer src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size)
    {
        void* mapped;
        auto [stagingBuf, stagingAlloc] = allocStaging(m_allocator, size, true, &mapped);

        uint64_t id = submitCopy(*m_queue, src, stagingBuf, srcOffset, 0, size);

        VmaAllocator allocator = m_allocator;
        return Future(m_queue.get(), id, [allocator, stagingBuf, stagingAlloc, mapped, dst, size]() {
            vmaInvalidateAllocation(allocator, stagingAlloc, 0, size);
            memcpy(dst, mapped, size);
            vmaDestroyBuffer(allocator, stagingBuf, stagingAlloc);
        });
    }
}