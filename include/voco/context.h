#pragma once
#include <vulkan/vulkan.h>

namespace voco
{
    struct Context
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    };
} // namespace voco