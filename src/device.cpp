#include <voco/device.h>
#include "queue.h"
#include "retirement_queue.h"
#include "descriptor_heap_backend.h"
#include "utils.h"
#include <spirv-reflect/spirv_reflect.h>
#include <cstring>
#include <fstream>
#include <sstream>
#ifdef VOCO_ENABLE_GLSL
#include <shaderc/shaderc.hpp>
#endif

namespace voco
{
    namespace
    {
        bool extensionEnabled(const std::vector<const char*>& list, const char* name)
        {
            for (const char* e : list)
                if (std::strcmp(e, name) == 0)
                    return true;
            return false;
        }

        // Best-effort fail-fast: Vulkan can only report supported features, not enabled
        // ones, so a supported-but-not-enabled feature still trips VK_CHECK later.
        void verifyRequiredFeatures(VkPhysicalDevice physicalDevice)
        {
            VkPhysicalDeviceDescriptorHeapFeaturesEXT featuresHeap{};
            featuresHeap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;

            VkPhysicalDeviceVulkan13Features features13{};
            features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            features13.pNext = &featuresHeap;

            VkPhysicalDeviceVulkan12Features features12{};
            features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            features12.pNext = &features13;

            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &features12;

            vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

            DEBUG_ASSERT(features12.timelineSemaphore,
                         "voco requires a device that supports timelineSemaphore");
            DEBUG_ASSERT(features13.synchronization2,
                         "voco requires a device that supports synchronization2");
            DEBUG_ASSERT(features12.bufferDeviceAddress,
                         "voco requires a device that supports bufferDeviceAddress");
            DEBUG_ASSERT(featuresHeap.descriptorHeap,
                         "voco requires a device that supports VK_EXT_descriptor_heap's descriptorHeap feature");
        }

        detail::HeapLimits queryDescriptorHeapLimits(VkPhysicalDevice physicalDevice)
        {
            VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps{};
            heapProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT;

            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &heapProps;

            vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

            return {
                heapProps.bufferDescriptorSize,
                heapProps.bufferDescriptorAlignment,
                heapProps.resourceHeapAlignment,
                heapProps.minResourceHeapReservedRange,
                heapProps.maxPushDataSize
            };
        }
    }

    Device::Device(const Context& context)
        : m_ctx(context)
    {
        verifyRequiredFeatures(m_ctx.physicalDevice);

        const bool wantHeap = extensionEnabled(m_ctx.enabledDeviceExtensions, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
        DEBUG_ASSERT(wantHeap, "voco requires VK_EXT_descriptor_heap to be enabled on the device");

        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = m_ctx.instance;
        allocatorInfo.physicalDevice = m_ctx.physicalDevice;
        allocatorInfo.device = m_ctx.device;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;
        // Every buffer voco creates now carries VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        // (needed for the descriptor heap), which VMA requires this flag for.
        allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

        vmaCreateAllocator(&allocatorInfo, &m_allocator);

        m_queue = std::make_unique<detail::Queue>(m_ctx.device, m_ctx.computeQueue, m_ctx.computeQueueFamilyIndex);
        m_retirementQueue = std::make_unique<detail::RetirementQueue>(*m_queue);

        m_heap = std::make_unique<detail::DescriptorHeapBackend>(
            m_ctx.device, m_allocator, queryDescriptorHeapLimits(m_ctx.physicalDevice), *m_retirementQueue);
    }

    Device::~Device()
    {
        // The retirement queue's destructor waits for all submitted GPU work to finish
        // and runs every pending callback -- including the heap backend's chunk-reclaim
        // callbacks, which capture a raw HeapChunk* back into m_heap -- before
        // returning. So m_heap must outlive that drain: destroy the retirement queue
        // first, then the heap backend can safely destroy every chunk unconditionally.
        m_retirementQueue.reset();
        m_heap.reset();
        m_queue.reset();

        vmaDestroyAllocator(m_allocator);
    }

    Buffer Device::createBuffer(BufferUsage usage, VkDeviceSize size, MemoryType memType)
    {
        auto memInfo = detail::toVmaMemoryInfo(memType);

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage) | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memInfo.usage;
        allocInfo.flags = memInfo.flags;

        VkBuffer vkBuffer;
        VmaAllocation allocation;
        VkResult res = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &vkBuffer, &allocation, nullptr);
        VK_CHECK(res);

        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = vkBuffer;
        VkDeviceAddress deviceAddress = vkGetBufferDeviceAddress(m_ctx.device, &addrInfo);

        return Buffer(m_allocator, m_retirementQueue.get(), vkBuffer, allocation, size, deviceAddress, usage, memType);
    }

    CommandList Device::createCommandList()
    {
        detail::TrackedCommandBuffer cmdBuf = m_queue->acquire();

        return CommandList(std::move(cmdBuf), m_queue.get(), m_heap.get(), &m_cacheMutex);
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

        uint32_t pushConstSize = 0;
        if (reflModule.push_constant_block_count > 0)
            pushConstSize = reflModule.push_constant_blocks[0].size;

        uint32_t bindingCount = 0;
        for (auto* reflSet : reflSets)
            bindingCount += reflSet->binding_count;

        const detail::HeapLimits& heapLimits = m_heap->limits();
        const uint32_t indexTableOffset = detail::alignUp(pushConstSize, 4u);
        DEBUG_ASSERT(indexTableOffset + bindingCount * sizeof(uint32_t) <= heapLimits.maxPushDataSize,
                     "shader has too many buffer bindings to fit voco's push-data index table");

        // Every (set,binding) maps to a push-data index -- the shader reads the actual
        // heap slot for a dispatch's binding from push data, not a fixed heap offset.
        // See src/descriptor_heap_backend.h for why a fixed offset can't work here.
        std::vector<ComputePipeline::BindingMapEntry> bindingMap;
        std::vector<VkDescriptorSetAndBindingMappingEXT> mappings;
        bindingMap.reserve(bindingCount);
        mappings.reserve(bindingCount);

        uint32_t nextPushOffset = indexTableOffset;
        for (auto* reflSet : reflSets)
        {
            for (uint32_t i = 0; i < reflSet->binding_count; ++i)
            {
                auto* b = reflSet->bindings[i];
                VkDescriptorType type = static_cast<VkDescriptorType>(b->descriptor_type);
                uint32_t pushOffset = nextPushOffset;
                nextPushOffset += static_cast<uint32_t>(sizeof(uint32_t));

                bindingMap.push_back({ reflSet->set, b->binding, pushOffset, type });

                VkDescriptorMappingSourcePushIndexEXT src{};
                src.heapOffset = 0; // indices are relative to whichever chunk is bound
                src.pushOffset = pushOffset;
                src.heapIndexStride = static_cast<uint32_t>(m_heap->slotStride());
                src.heapArrayStride = 0;
                src.pEmbeddedSampler = nullptr;
                src.useCombinedImageSamplerIndex = VK_FALSE;
                src.samplerHeapOffset = 0;
                src.samplerPushOffset = 0;
                src.samplerHeapIndexStride = 0;
                src.samplerHeapArrayStride = 0;

                VkDescriptorSetAndBindingMappingEXT mapping{};
                mapping.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT;
                mapping.descriptorSet = reflSet->set;
                mapping.firstBinding = b->binding;
                mapping.bindingCount = 1;
                mapping.resourceMask = (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                    ? VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT
                    : (VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT | VK_SPIRV_RESOURCE_TYPE_READ_WRITE_STORAGE_BUFFER_BIT_EXT);
                mapping.source = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
                mapping.sourceData.pushIndex = src;

                mappings.push_back(mapping);
            }
        }

        spvReflectDestroyShaderModule(&reflModule);

        VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{};
        mappingInfo.sType = VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT;
        mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
        mappingInfo.pMappings = mappings.data();

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
        shaderInfo.pCode = spirv.data();

        VkShaderModule shaderModule;
        VkResult res = vkCreateShaderModule(m_ctx.device, &shaderInfo, nullptr, &shaderModule);
        VK_CHECK(res);

        VkPipelineCreateFlags2CreateInfo flags2Info{};
        flags2Info.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
        flags2Info.flags = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &flags2Info;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.pNext = &mappingInfo;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = VK_NULL_HANDLE; // required for VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT

        VkPipeline pipeline;
        res = vkCreateComputePipelines(m_ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
        VK_CHECK(res);

        ComputePipeline result;
        result.m_device = m_ctx.device;
        result.m_retirementQueue = m_retirementQueue.get();
        result.m_shaderModule = shaderModule;
        result.m_pipeline = pipeline;
        result.m_bindingMap = std::move(bindingMap);
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

        VK_CHECK(vkEndCommandBuffer(cmd.m_cmdBuf->cmd));
        uint64_t submissionID = m_queue->submit(std::move(*cmd.m_cmdBuf));
        cmd.m_cmdBuf.reset();

        for (auto& bb : cmd.m_boundBuffers)
            bb.buffer->m_lastSubmissionID = submissionID;

        {
            std::lock_guard<std::mutex> cacheLock(m_cacheMutex);

            for (auto* pipeline : cmd.m_boundPipelines)
                pipeline->m_lastSubmissionID = submissionID;

            m_heap->onSubmitted(cmd.m_touchedChunks, submissionID);
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
        VK_CHECK(vkBeginCommandBuffer(cb.cmd, &beginInfo));

        VkBufferCopy region{};
        region.srcOffset = srcOffset;
        region.dstOffset = dstOffset;
        region.size = size;
        vkCmdCopyBuffer(cb.cmd, src, dst, 1, &region);

        VK_CHECK(vkEndCommandBuffer(cb.cmd));
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