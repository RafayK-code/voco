#pragma once
#include <vulkan/vulkan.h>
#include <string_view>
#include <cstdint>
#include <vector>

namespace voco
{
    namespace detail { class RetirementQueue; }

    class ComputePipeline
    {
    public:
        ~ComputePipeline();

        ComputePipeline(const ComputePipeline&) = delete;
        ComputePipeline& operator=(const ComputePipeline&) = delete;

        ComputePipeline(ComputePipeline&&) noexcept;
        ComputePipeline& operator=(ComputePipeline&&) noexcept;

        VkPipeline handle() const { return m_pipeline; }
        bool valid() const { return m_pipeline != VK_NULL_HANDLE; }

    private:
        friend class Device;
        friend class CommandList;

        // Where a (set,binding)'s heap slot index is pushed for the shader to read,
        // via HEAP_WITH_PUSH_INDEX_EXT mapping set up at pipeline creation.
        struct BindingMapEntry
        {
            uint32_t set = 0;
            uint32_t binding = 0;
            uint32_t pushOffset = 0;
            VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        };

        ComputePipeline() = default;

        void destroy();

        const BindingMapEntry* findBinding(uint32_t set, uint32_t binding) const;

        VkDevice m_device = VK_NULL_HANDLE;
        detail::RetirementQueue* m_retirementQueue = nullptr;
        uint64_t m_lastSubmissionID = 0;

        VkShaderModule m_shaderModule = VK_NULL_HANDLE;
        VkPipeline m_pipeline = VK_NULL_HANDLE;

        std::vector<BindingMapEntry> m_bindingMap;
    };
} // namespace voco
