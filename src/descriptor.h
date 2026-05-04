#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstdint>

namespace voco::detail
{
    struct BindingDesc
    {
        uint32_t binding;
        VkDescriptorType type;
        VkShaderStageFlags stages;

        bool operator==(const BindingDesc& other) const
        {
            return binding == other.binding && type == other.type && stages == other.stages;
        }
    };

    struct BindingDescVectorHash
    {
        size_t operator()(const std::vector<BindingDesc>& bindings) const;
    };

    struct DescriptorSetKey
    {
        std::vector<std::pair<uint32_t, VkBuffer>> bindings;

        bool operator==(const DescriptorSetKey& other) const;
    };

    struct CachedDescriptorSet
    {
        VkDescriptorSet set = VK_NULL_HANDLE;
        uint64_t lastSubmissionID = 0;
    };

    class DescriptorSetCache
    {
    public:
        DescriptorSetCache() = default;
        DescriptorSetCache(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout);
        ~DescriptorSetCache() = default;

        DescriptorSetCache(const DescriptorSetCache&) = delete;
        DescriptorSetCache& operator=(const DescriptorSetCache&) = delete;

        DescriptorSetCache(DescriptorSetCache&&) noexcept = default;
        DescriptorSetCache& operator=(DescriptorSetCache&&) noexcept = default;

        VkDescriptorSet get(const DescriptorSetKey& key, uint64_t lastFinishedID);
        VkDescriptorSet allocate(const DescriptorSetKey& key);
        void markSubmitted(const DescriptorSetKey& key, uint64_t submissionID);

    private:
        struct Hash
        {
            size_t operator()(const DescriptorSetKey& key) const;
        };

        VkDevice m_device = VK_NULL_HANDLE;
        VkDescriptorPool m_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;

        std::unordered_map<DescriptorSetKey, std::vector<CachedDescriptorSet>, Hash> m_cache;
    };

    struct DescriptorLayout
    {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        DescriptorSetCache setCache;
    };

    class DescriptorLayoutCache
    {
    public:
        DescriptorLayoutCache(VkDevice device, VkDescriptorPool pool);
        ~DescriptorLayoutCache();

        DescriptorLayoutCache(const DescriptorLayoutCache&) = delete;
        DescriptorLayoutCache& operator=(const DescriptorLayoutCache&) = delete;

        DescriptorLayout& getOrCreate(const std::vector<BindingDesc>& bindings);

    private:
        VkDevice m_device = VK_NULL_HANDLE;
        VkDescriptorPool m_pool = VK_NULL_HANDLE;
        std::unordered_map<std::vector<BindingDesc>, DescriptorLayout, BindingDescVectorHash> m_cache;
    };

    struct PipelineLayoutKey
    {
        std::vector<std::vector<BindingDesc>> setLayouts;
        std::optional<VkPushConstantRange> pushConstant;

        bool operator==(const PipelineLayoutKey& other) const;
    };

    class PipelineLayoutCache
    {
    public:
        PipelineLayoutCache(VkDevice device, DescriptorLayoutCache& descriptorCache);
        ~PipelineLayoutCache();

        PipelineLayoutCache(const PipelineLayoutCache&) = delete;
        PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

        VkPipelineLayout getOrCreate(const std::vector<std::vector<BindingDesc>>& setLayouts,
                                     const std::optional<VkPushConstantRange>& pushConstant);

    private:
        struct Hash
        {
            size_t operator()(const PipelineLayoutKey& key) const;
        };

        VkDevice m_device = VK_NULL_HANDLE;
        DescriptorLayoutCache& m_descriptorCache;
        std::unordered_map<PipelineLayoutKey, VkPipelineLayout, Hash> m_cache;
    };
}