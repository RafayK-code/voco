#pragma once
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include "types.h"
#include "buffer.h"
#include "pipeline.h"

namespace voco
{
    namespace detail
    {
        struct TrackedCommandBuffer;
        class DescriptorLayoutCache;
    }

    class CommandList
    {
    public:
        ~CommandList();

        CommandList(const CommandList&) = delete;
        CommandList& operator=(const CommandList&) = delete;

        CommandList(CommandList&&) noexcept = default;
        CommandList& operator=(CommandList&&) noexcept = default;

        void bindPipeline(ComputePipeline& pipeline);
        void bindBuffer(uint32_t set, uint32_t binding, Buffer& buffer, Access access = Access::ReadWrite);

        template<typename T>
        void setPushConstants(const T& data)
        {
            setPushConstantsImpl(&data, sizeof(T));
        }

        void dispatch(uint32_t x, uint32_t y, uint32_t z);

    private:
        friend class Device;

        CommandList(VkDevice device, detail::TrackedCommandBuffer cmdBuf,
                    detail::DescriptorLayoutCache* layoutCache, uint64_t lastFinishedID);

        void setPushConstantsImpl(const void* data, uint32_t size);

        VkDevice m_device = VK_NULL_HANDLE;
        detail::DescriptorLayoutCache* m_layoutCache = nullptr;
        uint64_t m_lastFinishedID = 0;

        std::unique_ptr<detail::TrackedCommandBuffer> m_cmdBuf;

        ComputePipeline* m_pipeline = nullptr;

        struct BoundBuffer
        {
            uint32_t set;
            uint32_t binding;
            Buffer* buffer;
            Access access;
        };

        std::vector<BoundBuffer> m_boundBuffers;
        std::unordered_map<uint32_t, std::vector<uint32_t>> m_setIndices;
    };
} // namespace voco
