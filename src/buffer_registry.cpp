#include "buffer_registry.h"
#include <algorithm>

namespace voco::detail
{
    BufferRegistry::BufferRegistry(DescriptorLayoutCache& layoutCache)
        : m_layoutCache(layoutCache)
    {}

    void BufferRegistry::registerRef(const std::vector<VkBuffer>& buffers, const std::vector<BindingDesc>& layoutKey)
    {
        for (VkBuffer buffer : buffers)
            m_bufferToLayouts[buffer].insert(layoutKey);
    }

    uint64_t BufferRegistry::onBufferDestroyed(VkBuffer buffer)
    {
        uint64_t maxSubmissionID = 0;

        if (auto extraIt = m_bufferExtraRetireID.find(buffer);
            extraIt != m_bufferExtraRetireID.end())
        {
            maxSubmissionID = std::max(maxSubmissionID, extraIt->second);
            m_bufferExtraRetireID.erase(extraIt);
        }

        auto it = m_bufferToLayouts.find(buffer);
        if (it == m_bufferToLayouts.end())
            return maxSubmissionID;

        for (const auto& layoutKey : it->second)
        {
            auto result = m_layoutCache.evictSetsContaining(layoutKey, buffer);
            maxSubmissionID = std::max(maxSubmissionID, result.maxSubmissionID);

            for (VkBuffer sibling : result.siblings)
                m_bufferExtraRetireID[sibling] = std::max(m_bufferExtraRetireID[sibling], result.maxSubmissionID);
        }

        m_bufferToLayouts.erase(it);
        return maxSubmissionID;
    }
}