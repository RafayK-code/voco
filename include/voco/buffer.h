#pragma once
#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>
#include "types.h"

namespace voco
{
    class Buffer
    {
    public:
        ~Buffer();

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        
        Buffer(Buffer&&) noexcept;
        Buffer& operator=(Buffer&&) noexcept;

        VkBuffer handle() const { return m_buffer; }
        VkDeviceSize size() const { return m_size; }
        MemoryType memoryType() const { return m_memoryType; }
        bool isHostVisible() const { return m_memoryType == MemoryType::Host || m_memoryType == MemoryType::Unified; }
        bool valid() const { return m_buffer != VK_NULL_HANDLE; }

    private:
        friend class Device;
        friend class CommandList;

        Buffer() = default;

        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        VkDeviceSize m_size = 0;

        BufferUsage m_usage = 0;
        MemoryType m_memoryType = MemoryType::Device;

        VkPipelineStageFlags2 m_lastStage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 m_lastAccess = VK_ACCESS_2_NONE;
    };
} // namespace voco
