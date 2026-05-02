#include <voco/pipeline.h>
#include "retirement_queue.h"

namespace voco
{
    ComputePipeline::~ComputePipeline()
    {
        if (m_pipeline == VK_NULL_HANDLE)
            return;

        VkDevice device = m_device;
        VkShaderModule shaderModule = m_shaderModule;
        VkPipelineLayout layout = m_pipelineLayout;
        VkPipeline pipeline = m_pipeline;

        m_retirementQueue->push({ m_lastSubmissionID, [device, shaderModule, layout, pipeline]() {
            vkDestroyShaderModule(device, shaderModule, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
        }});
    }

    ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
        : m_device(other.m_device)
        , m_retirementQueue(other.m_retirementQueue)
        , m_lastSubmissionID(other.m_lastSubmissionID)
        , m_shaderModule(other.m_shaderModule)
        , m_descSetLayouts(std::move(other.m_descSetLayouts))
        , m_pipelineLayout(other.m_pipelineLayout)
        , m_pipeline(other.m_pipeline)
        , m_pushConstSize(other.m_pushConstSize)
    {
        other.m_device = VK_NULL_HANDLE;
        other.m_retirementQueue = nullptr;
        other.m_shaderModule = VK_NULL_HANDLE;
        other.m_pipelineLayout = VK_NULL_HANDLE;
        other.m_pipeline = VK_NULL_HANDLE;
    }

    ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept
    {
        if (this == &other)
            return *this;

        this->~ComputePipeline();

        m_device = other.m_device;
        m_retirementQueue = other.m_retirementQueue;
        m_lastSubmissionID = other.m_lastSubmissionID;
        m_shaderModule = other.m_shaderModule;
        m_descSetLayouts = std::move(other.m_descSetLayouts);
        m_pipelineLayout = other.m_pipelineLayout;
        m_pipeline = other.m_pipeline;
        m_pushConstSize = other.m_pushConstSize;

        other.m_device = VK_NULL_HANDLE;
        other.m_retirementQueue = nullptr;
        other.m_shaderModule = VK_NULL_HANDLE;
        other.m_pipelineLayout = VK_NULL_HANDLE;
        other.m_pipeline = VK_NULL_HANDLE;

        return *this;
    }
}
