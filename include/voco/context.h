#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace voco
{
    struct Context
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // Extensions enabled at vkCreateDevice time (voco is BYOA, so it can only use
        // what's enabled here, not what the device merely supports). Also determines
        // the descriptor backend: VK_EXT_descriptor_heap opts into the heap backend.
        std::vector<const char*> enabledDeviceExtensions;
    };
} // namespace voco