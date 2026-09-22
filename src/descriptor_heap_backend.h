#pragma once
#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace voco::detail
{
    class RetirementQueue;

    // Sizes/alignments queried once from VkPhysicalDeviceDescriptorHeapPropertiesEXT.
    struct HeapLimits
    {
        VkDeviceSize bufferDescriptorSize = 0;
        VkDeviceSize bufferDescriptorAlignment = 0;
        VkDeviceSize resourceHeapAlignment = 0;
        VkDeviceSize minResourceHeapReservedRange = 0;
        VkDeviceSize maxPushDataSize = 0;
    };

    // One buffer binding resolved for an upcoming dispatch.
    struct DispatchBinding
    {
        uint32_t pushOffset; // where the shader reads this slot's index from, in push data
        VkDeviceAddress address;
        VkDeviceSize size;
        VkDescriptorType type;
    };

    // A host-visible, mapped VkBuffer holding a contiguous run of descriptor slots plus
    // the driver's reserved range. Owned exclusively by one CommandList between
    // acquisition and submission.
    struct HeapChunk
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        std::byte* mapped = nullptr;
        VkDeviceAddress baseAddress = 0;
        VkDeviceSize reservedRangeOffset = 0; // == capacitySlots * slotStride, aligned
        VkDeviceSize totalSize = 0;           // reservedRangeOffset + reservedRangeSize
        uint32_t capacitySlots = 0;
        uint32_t cursor = 0;
        uint64_t lastSubmissionID = 0;
    };

    // Descriptor management via VK_EXT_descriptor_heap. Descriptors are plain memory
    // voco writes fresh per dispatch from a buffer's live device address, so a slot's
    // valid window is bounded by the one submission that used it -- no reverse index
    // from VkBuffer to cached state is needed, unlike the old VkDescriptorSet cache.
    //
    // Slots are bump-allocated per resolved binding per dispatch, never rewritten in
    // place: voco allows rebinding different buffers to the same (set,binding) across
    // multiple dispatches within one CommandList, and all descriptor writes happen at
    // CPU record time, before any GPU execution -- a fixed slot rewritten per dispatch
    // would mean every dispatch reads the LAST host write, not its own.
    //
    // Chunks, not individual slots, are the reclaim unit, and a chunk is never shared
    // between two CommandLists -- each chunk's lastSubmissionID is therefore
    // unambiguous. A single shared bump cursor can't work here: slots are allocated at
    // record time, before a submission ID exists, and one CommandList can record
    // thousands of dispatches before it's ever submitted.
    //
    // allocateSlots assumes the caller holds Device's cache mutex. writeDescriptors and
    // bindHeap touch only the chunk/command buffer given to them.
    class DescriptorHeapBackend
    {
    public:
        DescriptorHeapBackend(VkDevice device, VmaAllocator allocator, HeapLimits limits,
                              RetirementQueue& retirementQueue);
        ~DescriptorHeapBackend();

        DescriptorHeapBackend(const DescriptorHeapBackend&) = delete;
        DescriptorHeapBackend& operator=(const DescriptorHeapBackend&) = delete;

        struct SlotAllocation
        {
            HeapChunk* chunk = nullptr;
            uint32_t baseSlot = 0;
        };

        // Bump-allocates `count` (> 0) contiguous slots. Reuses `activeChunk` if it has
        // room; otherwise acquires a fresh chunk (from the free pool, or newly created)
        // and returns that instead -- callers must compare the returned chunk against
        // `activeChunk` to know whether a rebind is needed.
        SlotAllocation allocateSlots(HeapChunk* activeChunk, uint32_t count);

        // Writes one resource descriptor per binding into consecutive slots starting at
        // baseSlot in `chunk`.
        void writeDescriptors(HeapChunk& chunk, uint32_t baseSlot, std::span<const DispatchBinding> bindings);

        void bindHeap(VkCommandBuffer cmd, const HeapChunk& chunk) const;

        // Wraps vkCmdPushDataEXT -- the only heap entry point CommandList needs
        // directly (for both the per-binding index table and user push constants),
        // so all VK_EXT_descriptor_heap function-pointer loading stays in this class.
        void pushData(VkCommandBuffer cmd, uint32_t offset, const void* data, size_t size) const;

        // Called from Device::submit, under the cache mutex. Stamps each touched
        // chunk's submission ID and queues its return to the free pool once that
        // submission retires.
        void onSubmitted(std::span<HeapChunk* const> touchedChunks, uint64_t submissionID);

        // Returns chunks from a CommandList that was never submitted -- the GPU never
        // saw them, so they go straight back to the free pool.
        void releaseChunks(std::span<HeapChunk* const> chunks);

        VkDeviceSize slotStride() const { return m_slotStride; }
        const HeapLimits& limits() const { return m_limits; }

    private:
        HeapChunk* acquireChunk(uint32_t neededSlots);
        HeapChunk* createChunk(uint32_t slots);

        VkDevice m_device = VK_NULL_HANDLE;
        VmaAllocator m_allocator = VK_NULL_HANDLE;
        HeapLimits m_limits;
        RetirementQueue& m_retirementQueue;

        VkDeviceSize m_slotStride = 0;

        // VK_EXT_descriptor_heap is provisional enough that the SDK's loader import
        // library has no direct trampolines for it yet (unlike core/promoted
        // functions) -- these must be resolved dynamically via vkGetDeviceProcAddr.
        PFN_vkWriteResourceDescriptorsEXT m_pfnWriteResourceDescriptors = nullptr;
        PFN_vkCmdBindResourceHeapEXT m_pfnCmdBindResourceHeap = nullptr;
        PFN_vkCmdPushDataEXT m_pfnCmdPushData = nullptr;

        std::vector<std::unique_ptr<HeapChunk>> m_allChunks;

        std::mutex m_freePoolMutex;
        std::vector<HeapChunk*> m_freePool;
    };
}
