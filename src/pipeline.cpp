#include <voco/pipeline.h>
#include "retirement_queue.h"

namespace voco
{
    ComputePipeline::~ComputePipeline()
    {
        destroy();
    }

    ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
        : m_device(other.m_device)
        , m_retirementQueue(other.m_retirementQueue)
        , m_lastSubmissionID(other.m_lastSubmissionID)
        , m_shaderModule(other.m_shaderModule)
        , m_pipeline(other.m_pipeline)
        , m_bindingMap(std::move(other.m_bindingMap))
    {
        other.m_device = VK_NULL_HANDLE;
        other.m_retirementQueue = nullptr;
        other.m_shaderModule = VK_NULL_HANDLE;
        other.m_pipeline = VK_NULL_HANDLE;
    }

    ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
    {
        if (this == &other)
            return *this;

        destroy();

        m_device = other.m_device;
        m_retirementQueue = other.m_retirementQueue;
        m_lastSubmissionID = other.m_lastSubmissionID;
        m_shaderModule = other.m_shaderModule;
        m_pipeline = other.m_pipeline;
        m_bindingMap = std::move(other.m_bindingMap);

        other.m_device = VK_NULL_HANDLE;
        other.m_retirementQueue = nullptr;
        other.m_shaderModule = VK_NULL_HANDLE;
        other.m_pipeline = VK_NULL_HANDLE;

        return *this;
    }

    const ComputePipeline::BindingMapEntry* ComputePipeline::findBinding(uint32_t set, uint32_t binding) const
    {
        for (const auto& entry : m_bindingMap)
            if (entry.set == set && entry.binding == binding)
                return &entry;

        return nullptr;
    }

    void ComputePipeline::destroy()
    {
        if (m_pipeline == VK_NULL_HANDLE)
            return;

        VkDevice device = m_device;
        VkShaderModule shaderModule = m_shaderModule;
        VkPipeline pipeline = m_pipeline;

        m_device = VK_NULL_HANDLE;
        m_shaderModule = VK_NULL_HANDLE;
        m_pipeline = VK_NULL_HANDLE;

        m_retirementQueue->push({ m_lastSubmissionID, [device, shaderModule, pipeline]() {
            vkDestroyShaderModule(device, shaderModule, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
        } });
    }
}
