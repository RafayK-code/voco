#pragma once
#include <vulkan/vulkan.h>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "types.h"
#include "buffer.h"
#include "pipeline.h"

namespace voco
{
    namespace detail
    {
        struct TrackedCommandBuffer;
        class Queue;
        class DescriptorHeapBackend;
        struct HeapChunk;
    }

    class CommandList
    {
    public:
        ~CommandList();

        CommandList(const CommandList&) = delete;
        CommandList& operator=(const CommandList&) = delete;

        CommandList(CommandList&&) noexcept = default;
        CommandList& operator=(CommandList&&) noexcept;

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

        CommandList(detail::TrackedCommandBuffer cmdBuf, detail::Queue* queue,
                    detail::DescriptorHeapBackend* heap, std::mutex* cacheMutex);

        void setPushConstantsImpl(const void* data, uint32_t size);

        // Returns the command buffer and heap chunks of a never-submitted list.
        void releaseUnsubmitted();

        detail::Queue* m_queue = nullptr;
        detail::DescriptorHeapBackend* m_heap = nullptr;
        std::mutex* m_cacheMutex = nullptr;
        detail::HeapChunk* m_activeChunk = nullptr;
        std::vector<detail::HeapChunk*> m_touchedChunks;

        std::unique_ptr<detail::TrackedCommandBuffer> m_cmdBuf;

        ComputePipeline* m_pipeline = nullptr;
        bool m_pipelineChanged = false;
        std::vector<ComputePipeline*> m_boundPipelines;

        struct BoundBuffer
        {
            uint32_t set;
            uint32_t binding;
            Buffer* buffer;
            Access access;
        };

        // Current buffer at a (set,binding). Persists across dispatches until rebound;
        // `dirty` marks bindings changed since the last dispatch.
        struct BoundBinding
        {
            uint32_t bufferIndex = 0;
            VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bool dirty = true;
        };

        std::vector<BoundBuffer> m_boundBuffers;
        std::map<std::pair<uint32_t, uint32_t>, BoundBinding> m_bindings;
    };
} // namespace voco