#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstdint>
#include "retirement_queue.h"

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
        DescriptorSetCache(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout, RetirementQueue* retirementQueue, std::mutex* descriptorPoolMutex);
        ~DescriptorSetCache();

        DescriptorSetCache(const DescriptorSetCache&) = delete;
        DescriptorSetCache& operator=(const DescriptorSetCache&) = delete;

        DescriptorSetCache(DescriptorSetCache&&) noexcept = default;
        DescriptorSetCache& operator=(DescriptorSetCache&&) noexcept = default;

        struct EvictionResult
        {
            uint64_t maxSubmissionID = 0;
            std::vector<VkBuffer> siblings;
        };

        VkDescriptorSet get(const DescriptorSetKey& key, uint64_t lastFinishedID);
        VkDescriptorSet allocate(const DescriptorSetKey& key);
        void markSubmitted(const DescriptorSetKey& key, uint64_t submissionID);
        uint64_t maxSubmissionID() const;
        uint64_t clear();
        EvictionResult evictSetsContaining(VkBuffer buffer);

    private:
        struct Hash
        {
            size_t operator()(const DescriptorSetKey& key) const;
        };

        VkDevice m_device = VK_NULL_HANDLE;
        VkDescriptorPool m_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
        RetirementQueue* m_retirementQueue = nullptr;
        std::mutex* m_descriptorPoolMutex = nullptr;

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
        DescriptorLayoutCache(VkDevice device, VkDescriptorPool pool, RetirementQueue& retirementQueue, std::mutex& descriptorPoolMutex);
        ~DescriptorLayoutCache();

        DescriptorLayoutCache(const DescriptorLayoutCache&) = delete;
        DescriptorLayoutCache& operator=(const DescriptorLayoutCache&) = delete;

        DescriptorLayout& getOrCreate(const std::vector<BindingDesc>& bindings);
        DescriptorSetCache::EvictionResult evictSetsContaining(const std::vector<BindingDesc>& bindings, VkBuffer buffer);

    private:
        VkDevice m_device = VK_NULL_HANDLE;
        VkDescriptorPool m_pool = VK_NULL_HANDLE;
        RetirementQueue& m_retirementQueue;
        std::mutex& m_descriptorPoolMutex;
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
        PipelineLayoutCache(VkDevice device, DescriptorLayoutCache& descriptorCache, RetirementQueue& retirementQueue);
        ~PipelineLayoutCache();

        PipelineLayoutCache(const PipelineLayoutCache&) = delete;
        PipelineLayoutCache& operator=(const PipelineLayoutCache&) = delete;

        VkPipelineLayout getOrCreate(const std::vector<std::vector<BindingDesc>>& setLayouts,
                                     const std::optional<VkPushConstantRange>& pushConstant);
        void markSubmitted(VkPipelineLayout layout, uint64_t submissionID);

    private:
        struct Hash
        {
            size_t operator()(const PipelineLayoutKey& key) const;
        };

        struct CachedLayout
        {
            VkPipelineLayout layout = VK_NULL_HANDLE;
            uint64_t lastSubmissionID = 0;
        };

        VkDevice m_device = VK_NULL_HANDLE;
        DescriptorLayoutCache& m_descriptorCache;
        RetirementQueue& m_retirementQueue;
        std::unordered_map<PipelineLayoutKey, CachedLayout, Hash> m_cache;
    };
}