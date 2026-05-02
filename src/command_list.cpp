#include <voco/command_list.h>
#include "queue.h"
#include "descriptor.h"
#include "utils.h"

namespace voco
{
    CommandList::CommandList(VkDevice device, detail::TrackedCommandBuffer cmdBuf,
                             detail::DescriptorLayoutCache* layoutCache, uint64_t lastFinishedID)
        : m_device(device)
        , m_layoutCache(layoutCache)
        , m_lastFinishedID(lastFinishedID)
        , m_cmdBuf(std::make_unique<detail::TrackedCommandBuffer>(std::move(cmdBuf)))
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(m_cmdBuf->cmd, &beginInfo));
    }

    CommandList::~CommandList() = default;

    void CommandList::bindPipeline(ComputePipeline& pipeline)
    {
        vkCmdBindPipeline(m_cmdBuf->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        m_pipeline = &pipeline;
    }

    void CommandList::bindBuffer(uint32_t set, uint32_t binding, Buffer& buffer, Access access)
    {
        uint32_t index = static_cast<uint32_t>(m_boundBuffers.size());
        m_boundBuffers.push_back({
            .set = set,
            .binding = binding,
            .buffer = &buffer,
            .access = access
        });
        m_setIndices[set].push_back(index);
    }

    void CommandList::dispatch(uint32_t x, uint32_t y, uint32_t z)
    {
        DEBUG_ASSERT(m_pipeline, "attempted to dispatch before binding pipeline");

        bool needsBarrier = false;
        for (auto& [set, indices] : m_setIndices)
        {
            for (uint32_t idx : indices)
            {
                if (m_boundBuffers[idx].buffer->m_lastAccess & VK_ACCESS_2_SHADER_WRITE_BIT)
                {
                    needsBarrier = true;
                    break;
                }
            }
            if (needsBarrier) break;
        }

        if (needsBarrier)
        {
            VkMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;

            vkCmdPipelineBarrier2(m_cmdBuf->cmd, &depInfo);
        }

        for (auto& [set, indices] : m_setIndices)
        {
            std::vector<detail::BindingDesc> bindings;
            bindings.reserve(indices.size());

            detail::DescriptorSetKey setKey;
            setKey.bindings.reserve(indices.size());

            for (uint32_t idx : indices)
            {
                BoundBuffer& bb = m_boundBuffers[idx];
                bool isUniform = static_cast<int>(bb.buffer->m_usage & BufferUsage::Uniform) != 0;

                bindings.push_back({
                    .binding = bb.binding,
                    .type = isUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .stages = VK_SHADER_STAGE_COMPUTE_BIT
                });

                setKey.bindings.push_back({ bb.binding, bb.buffer->handle() });
            }

            detail::DescriptorLayout& descLayout = m_layoutCache->getOrCreate(bindings);

            bool needsWrite = false;
            VkDescriptorSet descSet = descLayout.setCache.get(setKey, m_lastFinishedID);
            if (descSet == VK_NULL_HANDLE)
            {
                descSet = descLayout.setCache.allocate(setKey);
                needsWrite = true;
            }

            if (needsWrite)
            {
                std::vector<VkDescriptorBufferInfo> bufferInfos;
                bufferInfos.reserve(indices.size());

                std::vector<VkWriteDescriptorSet> writes;
                writes.reserve(indices.size());

                for (uint32_t idx : indices)
                {
                    BoundBuffer& bb = m_boundBuffers[idx];
                    bool isUniform = static_cast<int>(bb.buffer->m_usage & BufferUsage::Uniform) != 0;

                    VkDescriptorBufferInfo& bufInfo = bufferInfos.emplace_back();
                    bufInfo.buffer = bb.buffer->handle();
                    bufInfo.offset = 0;
                    bufInfo.range  = bb.buffer->size();

                    VkWriteDescriptorSet& write = writes.emplace_back();
                    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    write.dstSet          = descSet;
                    write.dstBinding      = bb.binding;
                    write.dstArrayElement = 0;
                    write.descriptorCount = 1;
                    write.descriptorType  = isUniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    write.pBufferInfo     = &bufferInfos.back();
                }

                vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }
            vkCmdBindDescriptorSets(m_cmdBuf->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->pipelineLayout(), set, 1, &descSet, 0, nullptr);
        }

        vkCmdDispatch(m_cmdBuf->cmd, x, y, z);

        for (auto& [set, indices] : m_setIndices)
        {
            for (uint32_t idx : indices)
            {
                BoundBuffer& bb = m_boundBuffers[idx];
                bb.buffer->m_lastAccess = detail::ConvertAccessToVulkanAccessFlags2(bb.access);
                bb.buffer->m_lastStage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            }
        }

        m_setIndices.clear();
    }

    void CommandList::setPushConstantsImpl(const void* data, uint32_t size)
    {
        DEBUG_ASSERT(m_pipeline, "attempted to set push constants before binding pipeline");
        vkCmdPushConstants(m_cmdBuf->cmd, m_pipeline->pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
    }
}
