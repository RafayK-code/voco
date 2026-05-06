#include "descriptor.h"
#include "utils.h"
#include <unordered_set>

namespace voco::detail
{
    size_t BindingDescVectorHash::operator()(const std::vector<BindingDesc>& bindings) const
    {
        size_t h = bindings.size();
        for (const auto& b : bindings)
        {
            h ^= std::hash<uint32_t>{}(b.binding) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(b.type)    + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(b.stages)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    bool DescriptorSetKey::operator==(const DescriptorSetKey& other) const
    {
        return bindings == other.bindings;
    }

    size_t DescriptorSetCache::Hash::operator()(const DescriptorSetKey& key) const
    {
        size_t h = key.bindings.size();
        for (const auto& [binding, buffer] : key.bindings)
        {
            h ^= std::hash<uint32_t>{}(binding) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(buffer)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    DescriptorSetCache::DescriptorSetCache(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout, RetirementQueue* retirementQueue, std::mutex* descriptorPoolMutex)
        : m_device(device)
        , m_pool(pool)
        , m_layout(layout)
        , m_retirementQueue(retirementQueue)
        , m_descriptorPoolMutex(descriptorPoolMutex)
    {}

    DescriptorSetCache::~DescriptorSetCache()
    {
        clear();
    }

    VkDescriptorSet DescriptorSetCache::get(const DescriptorSetKey& key, uint64_t lastFinishedID)
    {
        auto it = m_cache.find(key);
        if (it == m_cache.end())
            return VK_NULL_HANDLE;

        for (auto& entry : it->second)
            if (entry.lastSubmissionID <= lastFinishedID)
                return entry.set;

        return VK_NULL_HANDLE;
    }

    VkDescriptorSet DescriptorSetCache::allocate(const DescriptorSetKey& key)
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_layout;

        std::unique_lock<std::mutex> lock(*m_descriptorPoolMutex);

        VkDescriptorSet set;
        VkResult res = vkAllocateDescriptorSets(m_device, &allocInfo, &set);
        VK_CHECK(res);

        lock.unlock();

        m_cache[key].push_back({ set, 0 });
        return set;
    }

    void DescriptorSetCache::markSubmitted(const DescriptorSetKey& key, uint64_t submissionID)
    {
        auto it = m_cache.find(key);
        if (it == m_cache.end())
            return;

        for (auto& entry : it->second)
        {
            if (entry.lastSubmissionID == 0 || entry.lastSubmissionID < submissionID)
                entry.lastSubmissionID = submissionID;
        }
    }

    uint64_t DescriptorSetCache::maxSubmissionID() const
    {
        uint64_t max = 0;
        for (auto& [key, entries] : m_cache)
            for (auto& entry : entries)
                if (entry.lastSubmissionID > max)
                    max = entry.lastSubmissionID;
        return max;
    }

    uint64_t DescriptorSetCache::clear()
    {
        DEBUG_ASSERT(m_retirementQueue || m_cache.empty(), "DescriptorSetCache destroyed with sets but no retirement queue");

        std::vector<VkDescriptorSet> sets;
        uint64_t maxSubmissionID = 0;
        for (auto& [key, entries] : m_cache)
        {
            for (auto& entry : entries)
            {
                sets.push_back(entry.set);
                if (entry.lastSubmissionID > maxSubmissionID)
                    maxSubmissionID = entry.lastSubmissionID;
            }
        }

        m_cache.clear();

        if (sets.empty())
            return maxSubmissionID;

        VkDevice device = m_device;
        VkDescriptorPool pool = m_pool;
        m_retirementQueue->push({ maxSubmissionID, [device, pool, sets = std::move(sets), poolMutex = m_descriptorPoolMutex]() {
            std::lock_guard<std::mutex> lock(*poolMutex);
            vkFreeDescriptorSets(device, pool, static_cast<uint32_t>(sets.size()), sets.data());
        }});

        return maxSubmissionID;
    }

    DescriptorSetCache::EvictionResult DescriptorSetCache::evictSetsContaining(VkBuffer buffer)
    {
        EvictionResult result;
        std::vector<VkDescriptorSet> sets;
        std::unordered_set<VkBuffer> siblingSet;

        auto it = m_cache.begin();
        while (it != m_cache.end())
        {
            bool contains = false;
            for (const auto& [binding, handle] : it->first.bindings)
            {
                if (handle == buffer)
                {
                    contains = true;
                    break;
                }
            }

            if (!contains)
            {
                ++it;
                continue;
            }

            for (const auto& [binding, handle] : it->first.bindings)
            {
                if (handle != buffer)
                    siblingSet.insert(handle);
            }

            for (const auto& entry : it->second)
            {
                sets.push_back(entry.set);
                result.maxSubmissionID = std::max(result.maxSubmissionID, entry.lastSubmissionID);
            }

            it = m_cache.erase(it);
        }

        result.siblings.assign(siblingSet.begin(), siblingSet.end());

        if (!sets.empty())
        {
            VkDevice device = m_device;
            VkDescriptorPool pool = m_pool;
            m_retirementQueue->push({ result.maxSubmissionID, [device, pool, sets = std::move(sets), poolMutex = m_descriptorPoolMutex]() {
                std::lock_guard<std::mutex> lock(*poolMutex);
                vkFreeDescriptorSets(device, pool, static_cast<uint32_t>(sets.size()), sets.data());
            }});
        }

        return result;
    }

    DescriptorLayoutCache::DescriptorLayoutCache(VkDevice device, VkDescriptorPool pool, RetirementQueue& retirementQueue, std::mutex& descriptorPoolMutex)
        : m_device(device)
        , m_pool(pool)
        , m_retirementQueue(retirementQueue)
        , m_descriptorPoolMutex(descriptorPoolMutex)
    {}

    DescriptorLayoutCache::~DescriptorLayoutCache()
    {
        uint64_t maxID = 0;
        std::vector<VkDescriptorSetLayout> layouts;
        layouts.reserve(m_cache.size());

        for (auto& [key, entry] : m_cache)
        {
            layouts.push_back(entry.layout);
            maxID = std::max(maxID, entry.setCache.maxSubmissionID());
        }

        m_cache.clear();

        if (!layouts.empty())
        {
            VkDevice device = m_device;
            m_retirementQueue.push({ maxID, [device, layouts = std::move(layouts)]() {
                for (VkDescriptorSetLayout layout : layouts)
                    vkDestroyDescriptorSetLayout(device, layout, nullptr);
            }});
        }
    }

    DescriptorLayout& DescriptorLayoutCache::getOrCreate(const std::vector<BindingDesc>& bindings)
    {
        auto it = m_cache.find(bindings);
        if (it != m_cache.end())
            return it->second;

        std::vector<VkDescriptorSetLayoutBinding> vkBindings;
        vkBindings.reserve(bindings.size());

        for (const auto& b : bindings)
        {
            VkDescriptorSetLayoutBinding vkb{};
            vkb.binding = b.binding;
            vkb.descriptorType = b.type;
            vkb.descriptorCount = 1;
            vkb.stageFlags = b.stages;
            vkBindings.push_back(vkb);
        }

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = static_cast<uint32_t>(vkBindings.size());
        info.pBindings = vkBindings.data();

        VkDescriptorSetLayout layout;
        VkResult res = vkCreateDescriptorSetLayout(m_device, &info, nullptr, &layout);
        VK_CHECK(res);

        DescriptorSetCache setCache(m_device, m_pool, layout, &m_retirementQueue, &m_descriptorPoolMutex);

        auto [ins, ok] = m_cache.emplace(bindings, DescriptorLayout{layout, std::move(setCache)});
        return ins->second;
    }

    DescriptorSetCache::EvictionResult DescriptorLayoutCache::evictSetsContaining(const std::vector<BindingDesc>& bindings, VkBuffer buffer)
    {
        auto it = m_cache.find(bindings);
        if (it == m_cache.end())
            return {};

        return it->second.setCache.evictSetsContaining(buffer);
    }

    bool PipelineLayoutKey::operator==(const PipelineLayoutKey& other) const
    {
        if (setLayouts != other.setLayouts)
            return false;
        if (pushConstant.has_value() != other.pushConstant.has_value())
            return false;
        if (pushConstant.has_value())
        {
            const auto& a = *pushConstant;
            const auto& b = *other.pushConstant;
            return a.stageFlags == b.stageFlags && a.offset == b.offset && a.size == b.size;
        }
        return true;
    }

    size_t PipelineLayoutCache::Hash::operator()(const PipelineLayoutKey& key) const
    {
        BindingDescVectorHash setHash;
        size_t h = key.setLayouts.size();
        for (const auto& set : key.setLayouts)
            h ^= setHash(set) + 0x9e3779b9 + (h << 6) + (h >> 2);

        if (key.pushConstant)
        {
            h ^= std::hash<uint32_t>{}(key.pushConstant->stageFlags) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(key.pushConstant->offset)     + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(key.pushConstant->size)       + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    PipelineLayoutCache::PipelineLayoutCache(VkDevice device, DescriptorLayoutCache& descriptorCache, RetirementQueue& retirementQueue)
        : m_device(device)
        , m_descriptorCache(descriptorCache)
        , m_retirementQueue(retirementQueue)
    {}

    PipelineLayoutCache::~PipelineLayoutCache()
    {
        std::vector<VkPipelineLayout> layouts;
        uint64_t maxID = 0;
        layouts.reserve(m_cache.size());

        for (auto& [key, entry] : m_cache)
        {
            layouts.push_back(entry.layout);
            maxID = std::max(maxID, entry.lastSubmissionID);
        }

        m_cache.clear();

        if (!layouts.empty())
        {
            VkDevice device = m_device;
            m_retirementQueue.push({ maxID, [device, layouts = std::move(layouts)]() {
                for (VkPipelineLayout layout : layouts)
                    vkDestroyPipelineLayout(device, layout, nullptr);
            }});
        }
    }

    VkPipelineLayout PipelineLayoutCache::getOrCreate(const std::vector<std::vector<BindingDesc>>& setLayouts,
                                                      const std::optional<VkPushConstantRange>& pushConstant)
    {
        PipelineLayoutKey key{ setLayouts, pushConstant };

        auto it = m_cache.find(key);
        if (it != m_cache.end())
            return it->second.layout;

        std::vector<VkDescriptorSetLayout> vkLayouts;
        vkLayouts.reserve(setLayouts.size());
        for (const auto& bindings : setLayouts)
            vkLayouts.push_back(m_descriptorCache.getOrCreate(bindings).layout);

        VkPipelineLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount = static_cast<uint32_t>(vkLayouts.size());
        info.pSetLayouts = vkLayouts.data();

        if (pushConstant)
        {
            info.pushConstantRangeCount = 1;
            info.pPushConstantRanges = &*pushConstant;
        }

        VkPipelineLayout layout;
        VkResult res = vkCreatePipelineLayout(m_device, &info, nullptr, &layout);
        VK_CHECK(res);

        m_cache.emplace(std::move(key), CachedLayout{ layout, 0 });
        return layout;
    }

    void PipelineLayoutCache::markSubmitted(VkPipelineLayout layout, uint64_t submissionID)
    {
        for (auto& [key, entry] : m_cache)
        {
            if (entry.layout == layout)
            {
                entry.lastSubmissionID = std::max(entry.lastSubmissionID, submissionID);
                return;
            }
        }
    }
}