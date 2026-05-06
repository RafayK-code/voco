#include <voco/command_list.h>
#include <algorithm>
#include "queue.h"
#include "descriptor.h"
#include "buffer_registry.h"
#include "utils.h"

namespace voco
{
    CommandList::CommandList(VkDevice device, detail::TrackedCommandBuffer cmdBuf,
                             detail::DescriptorLayoutCache* layoutCache, detail::BufferRegistry* bufferRegistry,
                             uint64_t lastFinishedID)
        : m_device(device)
        , m_layoutCache(layoutCache)
        , m_bufferRegistry(bufferRegistry)
        , m_lastFinishedID(lastFinishedID)
        , m_cmdBuf(std::make_unique<detail::TrackedCommandBuffer>(std::move(cmdBuf)))
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkResult res = vkBeginCommandBuffer(m_cmdBuf->cmd, &beginInfo);
        VK_CHECK(res);
    }

    CommandList::~CommandList() = default;

    void CommandList::bindPipeline(ComputePipeline& pipeline)
    {
        vkCmdBindPipeline(m_cmdBuf->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        m_pipeline = &pipeline;
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

        auto& pending = m_pendingSets[set];

        pending.bindings[binding] = PendingBinding{
            .binding = binding,
            .bufferIndex = index,
            .type = isUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        };
    }

    void CommandList::dispatch(uint32_t x, uint32_t y, uint32_t z)
    {
        DEBUG_ASSERT(m_pipeline, "attempted to dispatch before binding pipeline");

        bool needsBarrier = false;

        for (auto& [set, pending] : m_pendingSets)
        {
            for (auto& [binding, pb] : pending.bindings)
            {
                BoundBuffer& bb = m_boundBuffers[pb.bufferIndex];

                if (bb.buffer->m_lastAccess & VK_ACCESS_2_SHADER_WRITE_BIT)
                {
                    needsBarrier = true;
                    break;
                }
            }

            if (needsBarrier)
                break;
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

        for (auto& [set, pending] : m_pendingSets)
        {
            std::vector<PendingBinding> ordered;
            ordered.reserve(pending.bindings.size());

            for (auto& [binding, pb] : pending.bindings)
                ordered.push_back(pb);

            std::sort(
                ordered.begin(),
                ordered.end(),
                [](const PendingBinding& a, const PendingBinding& b)
                {
                    return a.binding < b.binding;
                }
            );

            std::vector<detail::BindingDesc> bindings;
            bindings.reserve(ordered.size());

            detail::DescriptorSetKey setKey;
            setKey.bindings.reserve(ordered.size());

            for (const PendingBinding& pb : ordered)
            {
                BoundBuffer& bb = m_boundBuffers[pb.bufferIndex];

                bindings.push_back({
                    pb.binding,
                    pb.type,
                    VK_SHADER_STAGE_COMPUTE_BIT
                    });

                setKey.bindings.push_back({
                    pb.binding,
                    bb.buffer->handle()
                    });
            }

            detail::DescriptorLayout& descLayout =
                m_layoutCache->getOrCreate(bindings);

            bool needsWrite = false;

            VkDescriptorSet descSet =
                descLayout.setCache.get(setKey, m_lastFinishedID);

            if (descSet == VK_NULL_HANDLE)
            {
                descSet = descLayout.setCache.allocate(setKey);
                needsWrite = true;

                if (m_bufferRegistry)
                {
                    std::vector<VkBuffer> bufferHandles;
                    bufferHandles.reserve(ordered.size());

                    for (const PendingBinding& pb : ordered)
                    {
                        BoundBuffer& bb = m_boundBuffers[pb.bufferIndex];
                        bufferHandles.push_back(bb.buffer->handle());
                    }

                    m_bufferRegistry->registerRef(bufferHandles, bindings);
                }
            }

            if (needsWrite)
            {
                std::vector<VkDescriptorBufferInfo> bufferInfos(ordered.size());
                std::vector<VkWriteDescriptorSet> writes(ordered.size());

                for (size_t i = 0; i < ordered.size(); ++i)
                {
                    const PendingBinding& pb = ordered[i];
                    BoundBuffer& bb = m_boundBuffers[pb.bufferIndex];

                    bufferInfos[i].buffer = bb.buffer->handle();
                    bufferInfos[i].offset = 0;
                    bufferInfos[i].range = bb.buffer->size();

                    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[i].dstSet = descSet;
                    writes[i].dstBinding = pb.binding;
                    writes[i].dstArrayElement = 0;
                    writes[i].descriptorCount = 1;
                    writes[i].descriptorType = pb.type;
                    writes[i].pBufferInfo = &bufferInfos[i];
                }

                vkUpdateDescriptorSets(
                    m_device,
                    static_cast<uint32_t>(writes.size()),
                    writes.data(),
                    0,
                    nullptr
                );
            }

            vkCmdBindDescriptorSets(
                m_cmdBuf->cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                m_pipeline->pipelineLayout(),
                set,
                1,
                &descSet,
                0,
                nullptr
            );

            m_usedSets.push_back({
                &descLayout.setCache,
                setKey.bindings
                });
        }

        vkCmdDispatch(m_cmdBuf->cmd, x, y, z);

        for (auto& [set, pending] : m_pendingSets)
        {
            for (auto& [binding, pb] : pending.bindings)
            {
                BoundBuffer& bb = m_boundBuffers[pb.bufferIndex];

                bb.buffer->m_lastAccess =
                    detail::ConvertAccessToVulkanAccessFlags2(bb.access);

                bb.buffer->m_lastStage =
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
        }

        m_pendingSets.clear();
    }

    void CommandList::setPushConstantsImpl(const void* data, uint32_t size)
    {
        DEBUG_ASSERT(m_pipeline, "attempted to set push constants before binding pipeline");
        vkCmdPushConstants(m_cmdBuf->cmd, m_pipeline->pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
    }
}
