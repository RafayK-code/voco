#pragma once
#include <vulkan/vulkan.h>
#include <cassert>
#include <voco/types.h>

#ifdef NDEBUG
    #define DEBUG_ASSERT(expr, msg) ((void)0)
#else
    #define DEBUG_ASSERT(expr, msg) assert((expr) && (msg))
#endif

#define VK_CHECK(expr) do { VkResult _vk_res = (expr); DEBUG_ASSERT(_vk_res == VK_SUCCESS, "Vulkan call failed"); (void)_vk_res; } while(0)

namespace voco::detail
{
    template<typename T>
    inline constexpr T alignUp(T value, T alignment)
    {
        return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
    }

    inline constexpr VkAccessFlags2 ConvertAccessToVulkanAccessFlags2(Access access)
    {
        switch (access)
        {
        case Access::Read:      return VK_ACCESS_2_SHADER_READ_BIT;
        case Access::Write:     return VK_ACCESS_2_SHADER_WRITE_BIT_KHR;
        case Access::ReadWrite: return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        }

        return VK_ACCESS_2_NONE;
    }
}