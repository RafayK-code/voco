#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "descriptor.h"

namespace voco::detail
{
    class BufferRegistry
    {
    public:
        BufferRegistry(DescriptorLayoutCache& layoutCache);
        ~BufferRegistry() = default;

        BufferRegistry(const BufferRegistry&) = delete;
        BufferRegistry& operator=(const BufferRegistry&) = delete;

        void registerRef(const std::vector<VkBuffer>& buffers, const std::vector<BindingDesc>& layoutKey);
        uint64_t onBufferDestroyed(VkBuffer buffer);

    private:
        DescriptorLayoutCache& m_layoutCache;

        std::unordered_map<VkBuffer, std::unordered_set<std::vector<BindingDesc>, BindingDescVectorHash>> m_bufferToLayouts;
        std::unordered_map<VkBuffer, uint64_t> m_bufferExtraRetireID;
    };
}