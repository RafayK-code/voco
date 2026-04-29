#pragma once
#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>
#include <cstdint>
#include "utils.h"

namespace voco 
{
    enum class MemoryType : int
    {
        None = 0,
        Device,
        Host,
        Unified,
    };

    namespace detail
    {
        struct MemoryTypeInfo
        {
            VmaMemoryUsage usage;
            VmaAllocationCreateFlags flags;
        };

        constexpr MemoryTypeInfo toVmaMemoryInfo(MemoryType mem)
        {
            switch (mem)
            {
            case MemoryType::Device:  return { VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0 };
            case MemoryType::Host:    return { VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT };
            case MemoryType::Unified: return { VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT };
            default:                  return { VMA_MEMORY_USAGE_UNKNOWN, 0 };
            }
        }
    }

    enum class BufferUsage : int
    {
        None = 0,
        Storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        Uniform = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        TransferSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        TransferDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };

    VOCO_ENUM_CLASS_FLAG_OPERATORS(BufferUsage)

    enum class Access
    {
        None,
        Read,
        Write,
        ReadWrite,
    };
} // namespace voco
