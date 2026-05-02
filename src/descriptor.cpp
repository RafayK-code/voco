#include "descriptor.h"
#include "utils.h"

namespace voco::detail
{
    DescriptorLayoutCache::DescriptorLayoutCache(VkDevice device, VkDescriptorPool pool)
        : m_device(device)
        , m_pool(pool)
    {}

    DescriptorLayoutCache::~DescriptorLayoutCache()
    {
        for (auto& [key, entry] : m_cache)
            vkDestroyDescriptorSetLayout(m_device, entry.layout, nullptr);
    }

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
        VK_CHECK(vkCreateDescriptorSetLayout(m_device, &info, nullptr, &layout));

        DescriptorSetCache setCache(m_device, m_pool, layout);

        auto [ins, ok] = m_cache.emplace(bindings, DescriptorLayout{layout, std::move(setCache)});
        return ins->second;
    }

    bool DescriptorSetKey::operator==(const DescriptorSetKey& other) const
    {
        return bindings == other.bindings;
    }

    size_t DescriptorSetKeyHash::operator()(const DescriptorSetKey& key) const
    {
        size_t h = key.bindings.size();
        for (const auto& [binding, buffer] : key.bindings)
        {
            h ^= std::hash<uint32_t>{}(binding) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint64_t>{}(reinterpret_cast<uint64_t>(buffer)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    DescriptorSetCache::DescriptorSetCache(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout)
        : m_device(device)
        , m_pool(pool)
        , m_layout(layout)
    {}

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

        VkDescriptorSet set;
        VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &set));

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
            {
                entry.lastSubmissionID = submissionID;
                return;
            }
        }
    }
}
