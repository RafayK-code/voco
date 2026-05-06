#pragma once
#include <vulkan/vulkan.h>
#include <list>
#include <mutex>
#include <vector>
#include <cstdint>

namespace voco::detail
{
    struct TrackedCommandBuffer
    {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        uint64_t submissionID = 0;
    };

    class Queue
    {
    public:
        Queue(VkDevice device, VkQueue queue, uint32_t queueFamilyIndex);
        ~Queue();

        Queue(const Queue&) = delete;
        Queue& operator=(const Queue&) = delete;

        TrackedCommandBuffer acquire();
        uint64_t submit(TrackedCommandBuffer cb);

        // called on submit
        void retire();

        void wait(uint64_t value);

        uint64_t getLastSubmittedID() const { return m_lastSubmittedID; }
        uint64_t getLastFinishedID();

        VkSemaphore getTimelineSemaphore() const { return m_timelineSemaphore; }

        void addWaitSemaphore(VkSemaphore semaphore, uint64_t value);
        void addSignalSemaphore(VkSemaphore semaphore, uint64_t value);

    private:
        VkDevice m_device = VK_NULL_HANDLE;
        VkQueue m_queue = VK_NULL_HANDLE;
        uint32_t m_queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        VkSemaphore m_timelineSemaphore = VK_NULL_HANDLE;

        uint64_t m_lastSubmittedID = 0;

        std::list<TrackedCommandBuffer> m_inFlight;
        std::list<TrackedCommandBuffer> m_pool;

        std::mutex m_mutex;

        std::vector<VkSemaphore> m_waitSemaphores;
        std::vector<uint64_t> m_waitSemaphoreValues;
        std::vector<VkSemaphore> m_signalSemaphores;
        std::vector<uint64_t> m_signalSemaphoreValues;
    };
} // namespace voco::detail
