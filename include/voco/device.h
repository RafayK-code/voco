#pragma once
#include <vulkan/vulkan.h>
#include <memory>
#include <string_view>
#include "context.h"
#include "types.h"
#include "buffer.h"
#include "pipeline.h"
#include "command_list.h"

namespace voco
{
    struct TransferHandle
    {
        void wait();

    private:
        friend class Device;
        VkFence m_fence = VK_NULL_HANDLE;
        VkDevice m_device = VK_NULL_HANDLE;
    };

    class Device
    {
    public:
        Device(const Context& context);
        ~Device();

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        Buffer createBuffer(BufferUsage usage, VkDeviceSize size, MemoryType memType = MemoryType::Device);

        void copyToDevice(const void* src, Buffer& dst, VkDeviceSize size);
        TransferHandle copyToDeviceAsync(const void* src, Buffer& dst, VkDeviceSize size);

        void copyToHost(const Buffer& src, void* dst, VkDeviceSize size);
        TransferHandle copyToHostAsync(const Buffer& src, void* dst, VkDeviceSize size);

        ComputePipeline createComputePipeline(std::string_view shaderPath);
        CommandList createCommandList();
        void submit(CommandList& cmd);

    private:
        void copyToDeviceMapped(const void*, VkBuffer, VmaAllocation, VkDeviceSize);
        void copyToDeviceStaged(const void*, VkBuffer, VkDeviceSize);
        void copyToHostMapped(VkBuffer, VmaAllocation, void*, VkDeviceSize);
        void copyToHostStaged(VkBuffer, void*, VkDeviceSize);

        TransferHandle copyToDeviceMappedAsync(const void*, VkBuffer, VmaAllocation, VkDeviceSize);
        TransferHandle copyToDeviceStagedAsync(const void*, VkBuffer, VkDeviceSize);
        TransferHandle copyToHostMappedAsync(VkBuffer, VmaAllocation, void*, VkDeviceSize);
        TransferHandle copyToHostStagedAsync(VkBuffer, void*, VkDeviceSize);

        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
} // namespace voco
