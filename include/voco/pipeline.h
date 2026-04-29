#pragma once
#include <vulkan/vulkan.h>
#include <string_view>
#include <cstdint>
#include <vector>

namespace voco
{
    class ComputePipeline
    {
    public:
        ~ComputePipeline();

        ComputePipeline(const ComputePipeline&) = delete;
        ComputePipeline& operator=(const ComputePipeline&) = delete;

        ComputePipeline(ComputePipeline&&) noexcept;
        ComputePipeline& operator=(ComputePipeline&&) noexcept;

        VkPipeline handle() const { return m_pipeline; }
        VkPipelineLayout pipelineLayout() const { return m_pipelineLayout; }
        bool valid() const { return m_pipeline != VK_NULL_HANDLE; }

    private:
        friend class Device;
        friend class CommandList;

        ComputePipeline() = default;

        VkDevice m_device = VK_NULL_HANDLE;
        VkShaderModule m_shaderModule = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> m_descSetLayouts;
        VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;
        uint32_t m_pushConstSize = 0;
    };
} // namespace voco
