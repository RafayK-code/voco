#pragma once
#include <vulkan/vulkan.h>
#include <memory>
#include <string_view>
#include "context.h"
#include "types.h"
#include "buffer.h"
#include "pipeline.h"
#include "command_list.h"
#include "future.h"

namespace voco
{
    namespace detail
    {
        class Queue;
        class RetirementQueue;
        class DescriptorLayoutCache;
        class PipelineLayoutCache;
    }

    class Device
    {
    public:
        Device(const Context& context);
        ~Device();

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        Buffer createBuffer(BufferUsage usage, VkDeviceSize size, MemoryType memType = MemoryType::Device);

        void copyToDevice(const void* src, Buffer& dst, VkDeviceSize offset, VkDeviceSize size);
        Future copyToDeviceAsync(const void* src, Buffer& dst, VkDeviceSize offset, VkDeviceSize size);

        void copyToHost(Buffer& src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size);
        Future copyToHostAsync(Buffer& src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size);

        ComputePipeline createComputePipeline(std::string_view shaderPath, ShaderSourceType sourceType = ShaderSourceType::SPIRV);
        CommandList createCommandList();
        void submit(CommandList& cmd);

    private:
        void copyToDeviceMapped(const void* src, VkBuffer dst, VmaAllocation allocation, VkDeviceSize dstOffset, VkDeviceSize size);
        void copyToDeviceStaged(const void* src, VkBuffer dst, VkDeviceSize dstOffset, VkDeviceSize size);
        void copyToHostMapped(VkBuffer src, VmaAllocation allocation, void* dst, VkDeviceSize srcOffset, VkDeviceSize size);
        void copyToHostStaged(VkBuffer src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size);

        Future copyToDeviceMappedAsync(const void* src, VkBuffer dst, VmaAllocation allocation, VkDeviceSize dstOffset, VkDeviceSize size);
        Future copyToDeviceStagedAsync(const void* src, VkBuffer dst, VkDeviceSize dstOffset, VkDeviceSize size);
        Future copyToHostMappedAsync(VkBuffer src, VmaAllocation allocation, void* dst, VkDeviceSize srcOffset, VkDeviceSize size);
        Future copyToHostStagedAsync(VkBuffer src, void* dst, VkDeviceSize srcOffset, VkDeviceSize size);

        Context m_ctx;
        VmaAllocator m_allocator = VK_NULL_HANDLE;
        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

        std::unique_ptr<detail::Queue> m_queue;
        std::unique_ptr<detail::DescriptorLayoutCache> m_descriptorLayoutCache;
        std::unique_ptr<detail::PipelineLayoutCache> m_pipelineLayoutCache;
        std::unique_ptr<detail::RetirementQueue> m_retirementQueue;
    };
} // namespace voco
