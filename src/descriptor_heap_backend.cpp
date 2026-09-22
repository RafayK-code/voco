#include "descriptor_heap_backend.h"
#include "retirement_queue.h"
#include "utils.h"
#include <algorithm>

namespace voco::detail
{
    namespace
    {
        constexpr uint32_t kDefaultChunkSlots = 4096;
    }

    DescriptorHeapBackend::DescriptorHeapBackend(VkDevice device, VmaAllocator allocator, HeapLimits limits,
                                                 RetirementQueue& retirementQueue)
        : m_device(device)
        , m_allocator(allocator)
        , m_limits(limits)
        , m_retirementQueue(retirementQueue)
        , m_slotStride(alignUp(limits.bufferDescriptorSize, limits.bufferDescriptorAlignment))
    {
        // VK_EXT_descriptor_heap is provisional enough that vulkan-1.lib has no direct
        // trampolines for it yet -- resolve these dynamically instead of linking them.
        m_pfnWriteResourceDescriptors = reinterpret_cast<PFN_vkWriteResourceDescriptorsEXT>(
            vkGetDeviceProcAddr(device, "vkWriteResourceDescriptorsEXT"));
        m_pfnCmdBindResourceHeap = reinterpret_cast<PFN_vkCmdBindResourceHeapEXT>(
            vkGetDeviceProcAddr(device, "vkCmdBindResourceHeapEXT"));
        m_pfnCmdPushData = reinterpret_cast<PFN_vkCmdPushDataEXT>(
            vkGetDeviceProcAddr(device, "vkCmdPushDataEXT"));

        DEBUG_ASSERT(m_pfnWriteResourceDescriptors && m_pfnCmdBindResourceHeap && m_pfnCmdPushData,
                     "driver reported VK_EXT_descriptor_heap support but is missing an entry point");
    }

    DescriptorHeapBackend::~DescriptorHeapBackend()
    {
        // Safe to destroy every chunk unconditionally: Device::~Device destroys the
        // retirement queue before this backend, which waits for all submitted GPU work
        // to finish and runs every pending reclaim callback before returning -- so no
        // chunk can still be in flight or referenced by a pending callback here.
        for (auto& chunk : m_allChunks)
            vmaDestroyBuffer(m_allocator, chunk->buffer, chunk->allocation);
    }

    HeapChunk* DescriptorHeapBackend::createChunk(uint32_t slots)
    {
        VkDeviceSize dataBytes = VkDeviceSize(slots) * m_slotStride;
        VkDeviceSize reservedOffset = alignUp(dataBytes, m_limits.resourceHeapAlignment);
        VkDeviceSize totalSize = reservedOffset + m_limits.minResourceHeapReservedRange;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = totalSize;
        bufferInfo.usage = VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocResult{};
        VkResult res = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &buffer, &allocation, &allocResult);
        VK_CHECK(res);

        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        VkDeviceAddress baseAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);

        auto chunk = std::make_unique<HeapChunk>();
        chunk->buffer = buffer;
        chunk->allocation = allocation;
        chunk->mapped = static_cast<std::byte*>(allocResult.pMappedData);
        chunk->baseAddress = baseAddress;
        chunk->reservedRangeOffset = reservedOffset;
        chunk->totalSize = totalSize;
        chunk->capacitySlots = slots;

        HeapChunk* raw = chunk.get();
        m_allChunks.push_back(std::move(chunk));
        return raw;
    }

    HeapChunk* DescriptorHeapBackend::acquireChunk(uint32_t neededSlots)
    {
        {
            std::lock_guard<std::mutex> lock(m_freePoolMutex);

            auto it = std::find_if(m_freePool.begin(), m_freePool.end(),
                [neededSlots](HeapChunk* c) { return c->capacitySlots >= neededSlots; });

            if (it != m_freePool.end())
            {
                HeapChunk* chunk = *it;
                m_freePool.erase(it);
                chunk->cursor = 0;
                return chunk;
            }
        }

        return createChunk(std::max(kDefaultChunkSlots, neededSlots));
    }

    DescriptorHeapBackend::SlotAllocation DescriptorHeapBackend::allocateSlots(HeapChunk* activeChunk, uint32_t count)
    {
        if (activeChunk && activeChunk->cursor + count <= activeChunk->capacitySlots)
        {
            uint32_t base = activeChunk->cursor;
            activeChunk->cursor += count;
            return { activeChunk, base };
        }

        HeapChunk* chunk = acquireChunk(count);
        uint32_t base = chunk->cursor;
        chunk->cursor += count;
        return { chunk, base };
    }

    void DescriptorHeapBackend::writeDescriptors(HeapChunk& chunk, uint32_t baseSlot, std::span<const DispatchBinding> bindings)
    {
        std::vector<VkDeviceAddressRangeEXT> ranges(bindings.size());
        std::vector<VkResourceDescriptorInfoEXT> infos(bindings.size());
        std::vector<VkHostAddressRangeEXT> dsts(bindings.size());

        for (size_t i = 0; i < bindings.size(); ++i)
        {
            ranges[i] = { bindings[i].address, bindings[i].size };

            infos[i] = {};
            infos[i].sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT;
            infos[i].type = bindings[i].type;
            infos[i].data.pAddressRange = &ranges[i];

            dsts[i].address = chunk.mapped + (baseSlot + i) * m_slotStride;
            dsts[i].size = static_cast<size_t>(m_limits.bufferDescriptorSize);
        }

        VK_CHECK(m_pfnWriteResourceDescriptors(m_device, static_cast<uint32_t>(infos.size()), infos.data(), dsts.data()));
    }

    void DescriptorHeapBackend::bindHeap(VkCommandBuffer cmd, const HeapChunk& chunk) const
    {
        VkBindHeapInfoEXT bindInfo{};
        bindInfo.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT;
        bindInfo.heapRange = { chunk.baseAddress, chunk.totalSize };
        bindInfo.reservedRangeOffset = chunk.reservedRangeOffset;
        bindInfo.reservedRangeSize = m_limits.minResourceHeapReservedRange;

        m_pfnCmdBindResourceHeap(cmd, &bindInfo);
    }

    void DescriptorHeapBackend::pushData(VkCommandBuffer cmd, uint32_t offset, const void* data, size_t size) const
    {
        VkPushDataInfoEXT pushInfo{};
        pushInfo.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT;
        pushInfo.offset = offset;
        pushInfo.data.address = data;
        pushInfo.data.size = size;

        m_pfnCmdPushData(cmd, &pushInfo);
    }

    void DescriptorHeapBackend::onSubmitted(std::span<HeapChunk* const> touchedChunks, uint64_t submissionID)
    {
        for (HeapChunk* chunk : touchedChunks)
        {
            chunk->lastSubmissionID = submissionID;

            m_retirementQueue.push({ submissionID, [this, chunk]() {
                std::lock_guard<std::mutex> lock(m_freePoolMutex);
                m_freePool.push_back(chunk);
            }});
        }
    }

    void DescriptorHeapBackend::releaseChunks(std::span<HeapChunk* const> chunks)
    {
        std::lock_guard<std::mutex> lock(m_freePoolMutex);
        m_freePool.insert(m_freePool.end(), chunks.begin(), chunks.end());
    }
}
