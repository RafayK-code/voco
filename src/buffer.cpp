#include <voco/buffer.h>
#include "retirement_queue.h"
#include "buffer_registry.h"

namespace voco
{
    Buffer::Buffer(VmaAllocator allocator, detail::RetirementQueue* retirementQueue,
                   detail::BufferRegistry* bufferRegistry, VkBuffer buffer,
                   VmaAllocation allocation, VkDeviceSize size, BufferUsage usage, MemoryType memType)
        : m_retirementQueue(retirementQueue)
        , m_bufferRegistry(bufferRegistry)
        , m_allocator(allocator)
        , m_buffer(buffer)
        , m_allocation(allocation)
        , m_size(size)
        , m_usage(usage)
        , m_memoryType(memType)
    {}

    Buffer::~Buffer()
    {
        destroy();
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : m_retirementQueue(other.m_retirementQueue)
        , m_bufferRegistry(other.m_bufferRegistry)
        , m_allocator(other.m_allocator)
        , m_buffer(other.m_buffer)
        , m_allocation(other.m_allocation)
        , m_size(other.m_size)
        , m_usage(other.m_usage)
        , m_memoryType(other.m_memoryType)
        , m_lastSubmissionID(other.m_lastSubmissionID)
        , m_lastStage(other.m_lastStage)
        , m_lastAccess(other.m_lastAccess)
    {
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_retirementQueue = nullptr;
        other.m_bufferRegistry = nullptr;
        other.m_allocator = VK_NULL_HANDLE;
    }

    Buffer& Buffer::operator=(Buffer&& other) noexcept
    {
        if (this == &other)
            return *this;

        destroy();

        m_retirementQueue = other.m_retirementQueue;
        m_bufferRegistry = other.m_bufferRegistry;
        m_allocator = other.m_allocator;
        m_buffer = other.m_buffer;
        m_allocation = other.m_allocation;
        m_size = other.m_size;
        m_usage = other.m_usage;
        m_memoryType = other.m_memoryType;
        m_lastSubmissionID = other.m_lastSubmissionID;
        m_lastStage = other.m_lastStage;
        m_lastAccess = other.m_lastAccess;

        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_retirementQueue = nullptr;
        other.m_bufferRegistry = nullptr;
        other.m_allocator = VK_NULL_HANDLE;

        return *this;
    }

    void Buffer::destroy()
    {
        if (m_buffer == VK_NULL_HANDLE)
            return;

        VkBuffer buffer = m_buffer;
        VmaAllocation allocation = m_allocation;
        VmaAllocator allocator = m_allocator;

        m_buffer = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
        m_allocator = VK_NULL_HANDLE;

        uint64_t descriptorMaxSubmissionID = 0;
        if (m_bufferRegistry)
            descriptorMaxSubmissionID = m_bufferRegistry->onBufferDestroyed(buffer);

        uint64_t lastSubmissionID = std::max(m_lastSubmissionID, descriptorMaxSubmissionID);
        if (lastSubmissionID == 0 || m_retirementQueue == nullptr)
        {
            vmaDestroyBuffer(allocator, buffer, allocation);
            return;
        }

        m_retirementQueue->push({ lastSubmissionID, [allocator, buffer, allocation]() {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }});
    }
}