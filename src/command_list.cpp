#include <voco/command_list.h>
#include "queue.h"
#include "descriptor_heap_backend.h"
#include "utils.h"

namespace voco
{
    CommandList::CommandList(detail::TrackedCommandBuffer cmdBuf, detail::Queue* queue,
                             detail::DescriptorHeapBackend* heap, std::mutex* cacheMutex)
        : m_queue(queue)
        , m_heap(heap)
        , m_cacheMutex(cacheMutex)
        , m_cmdBuf(std::make_unique<detail::TrackedCommandBuffer>(std::move(cmdBuf)))
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkResult res = vkBeginCommandBuffer(m_cmdBuf->cmd, &beginInfo);
        VK_CHECK(res);
    }

    CommandList::~CommandList()
    {
        releaseUnsubmitted();
    }

    CommandList& CommandList::operator=(CommandList&& other) noexcept
    {
        if (this == &other)
            return *this;

        releaseUnsubmitted();

        m_queue = other.m_queue;
        m_heap = other.m_heap;
        m_cacheMutex = other.m_cacheMutex;
        m_activeChunk = other.m_activeChunk;
        m_touchedChunks = std::move(other.m_touchedChunks);
        m_cmdBuf = std::move(other.m_cmdBuf);
        m_pipeline = other.m_pipeline;
        m_pipelineChanged = other.m_pipelineChanged;
        m_boundPipelines = std::move(other.m_boundPipelines);
        m_boundBuffers = std::move(other.m_boundBuffers);
        m_bindings = std::move(other.m_bindings);

        other.m_activeChunk = nullptr;
        other.m_touchedChunks.clear();

        return *this;
    }

    void CommandList::releaseUnsubmitted()
    {
        // Device::submit takes m_cmdBuf, so a live one means this list never reached
        // the GPU -- its command buffer and heap chunks can be recycled immediately.
        if (!m_cmdBuf)
            return;

        m_queue->release(std::move(*m_cmdBuf));
        m_cmdBuf.reset();

        m_heap->releaseChunks(m_touchedChunks);
        m_touchedChunks.clear();
        m_activeChunk = nullptr;
    }

    void CommandList::bindPipeline(ComputePipeline& pipeline)
    {
        vkCmdBindPipeline(m_cmdBuf->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        m_pipeline = &pipeline;
        m_pipelineChanged = true;
        m_boundPipelines.push_back(&pipeline);
    }

    void CommandList::bindBuffer(uint32_t set, uint32_t binding, Buffer& buffer, Access access)
    {
        const uint32_t index = static_cast<uint32_t>(m_boundBuffers.size());

        m_boundBuffers.push_back({
            .set = set,
            .binding = binding,
            .buffer = &buffer,
            .access = access
            });

        const bool isUniform =
            static_cast<int>(buffer.m_usage & BufferUsage::Uniform) != 0;

        m_bindings[{ set, binding }] = BoundBinding{
            .bufferIndex = index,
            .type = isUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .dirty = true
        };
    }

    void CommandList::dispatch(uint32_t x, uint32_t y, uint32_t z)
    {
        DEBUG_ASSERT(m_pipeline, "attempted to dispatch before binding pipeline");

        // Every binding this dispatch reads, including ones left bound from earlier
        // dispatches. Only dirty ones need new descriptors -- unless the pipeline
        // changed, since each pipeline reads its heap indices from its own push-data
        // offsets, which nothing has written yet.
        std::vector<BoundBuffer*> used;
        std::vector<detail::DispatchBinding> resolved;
        used.reserve(m_bindings.size());
        resolved.reserve(m_bindings.size());

        for (auto& [key, bound] : m_bindings)
        {
            const auto [set, binding] = key;
            const ComputePipeline::BindingMapEntry* entry = m_pipeline->findBinding(set, binding);
            if (!entry)
            {
                // Stale bindings left over from a previous pipeline are fine to skip.
                DEBUG_ASSERT(!bound.dirty, "bound buffer has no matching binding in the shader's reflected layout");
                continue;
            }

            BoundBuffer& bb = m_boundBuffers[bound.bufferIndex];
            used.push_back(&bb);

            if (bound.dirty || m_pipelineChanged)
                resolved.push_back({ entry->pushOffset, bb.buffer->deviceAddress(), bb.buffer->size(), bound.type });
        }

        bool needsBarrier = false;
        for (BoundBuffer* bb : used)
        {
            if (bb->buffer->m_lastAccess & VK_ACCESS_2_SHADER_WRITE_BIT)
            {
                needsBarrier = true;
                break;
            }
        }

        if (needsBarrier)
        {
            VkMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask =
                VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_SHADER_WRITE_BIT;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;

            vkCmdPipelineBarrier2(m_cmdBuf->cmd, &depInfo);
        }

        if (!resolved.empty())
        {
            detail::DescriptorHeapBackend::SlotAllocation alloc;
            {
                std::lock_guard<std::mutex> cacheLock(*m_cacheMutex);
                alloc = m_heap->allocateSlots(m_activeChunk, static_cast<uint32_t>(resolved.size()));
            }

            m_heap->writeDescriptors(*alloc.chunk, alloc.baseSlot, resolved);

            if (alloc.chunk != m_activeChunk)
            {
                m_heap->bindHeap(m_cmdBuf->cmd, *alloc.chunk);
                m_activeChunk = alloc.chunk;
                m_touchedChunks.push_back(alloc.chunk);
            }

            // One small push per binding rather than a batched range: each binding's
            // push-data offset was assigned independently at pipeline creation (see
            // device.cpp), so the offsets touched by this dispatch aren't guaranteed
            // contiguous.
            for (size_t i = 0; i < resolved.size(); ++i)
            {
                uint32_t index = alloc.baseSlot + static_cast<uint32_t>(i);
                m_heap->pushData(m_cmdBuf->cmd, resolved[i].pushOffset, &index, sizeof(index));
            }
        }

        vkCmdDispatch(m_cmdBuf->cmd, x, y, z);

        for (BoundBuffer* bb : used)
        {
            bb->buffer->m_lastAccess =
                detail::ConvertAccessToVulkanAccessFlags2(bb->access);

            bb->buffer->m_lastStage =
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        }

        for (auto& [key, bound] : m_bindings)
            bound.dirty = false;

        m_pipelineChanged = false;
    }

    void CommandList::setPushConstantsImpl(const void* data, uint32_t size)
    {
        DEBUG_ASSERT(m_pipeline, "attempted to set push constants before binding pipeline");
        // offset 0 matches the reflected push-constant block's offset (always 0 today)
        m_heap->pushData(m_cmdBuf->cmd, 0, data, size);
    }
}
