#include <voco/buffer.h>
#include "retirement_queue.h"

namespace voco
{
    Buffer::Buffer(VmaAllocator allocator, detail::RetirementQueue* retirementQueue, VkBuffer buffer,
                   VmaAllocation allocation, VkDeviceSize size, BufferUsage usage, MemoryType memType)
        : m_retirementQueue(retirementQueue)
        , m_allocator(allocator)
        , m_buffer(buffer)
        , m_allocation(allocation)
        , m_size(size)
        , m_usage(usage)
        , m_memoryType(memType)
    {}

    Buffer::~Buffer()
    {
        if (m_buffer == VK_NULL_HANDLE)
            return;

        if (m_lastSubmissionID == 0 || m_retirementQueue == nullptr)
        {
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
            return;
        }

        m_retirementQueue->push({
            .submissionID = m_lastSubmissionID,
            .destroy = [allocator = m_allocator, buffer = m_buffer, allocation = m_allocation]() {
                vmaDestroyBuffer(allocator, buffer, allocation);
            }
        });
    }

    Buffer::Buffer(Buffer&& other) noexcept
        : m_retirementQueue(other.m_retirementQueue)
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
        other.m_allocator = VK_NULL_HANDLE;
    }

    Buffer& Buffer::operator=(Buffer&& other) noexcept
    {
        if (this == &other)
            return *this;

        this->~Buffer();

        m_retirementQueue = other.m_retirementQueue;
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
        other.m_allocator = VK_NULL_HANDLE;

        return *this;
    }
}