#pragma once
#include <vulkan/vulkan.h>
#include <functional>
#include <cstdint>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "queue.h"

namespace voco::detail
{
    struct PendingDestroy
    {
        uint64_t submissionID;
        std::function<void()> destroy;
    };

    class RetirementQueue
    {
    public:
        RetirementQueue(Queue& queue);
        ~RetirementQueue();

        RetirementQueue(const RetirementQueue&) = delete;
        RetirementQueue& operator=(const RetirementQueue&) = delete;

        void push(PendingDestroy pending);

    private:
        void retirementLoop();

        Queue& m_queue;

        std::deque<PendingDestroy> m_pending;

        std::mutex m_retirementMutex;
        std::condition_variable m_retirementCV;
        std::thread m_retirementThread;
        std::atomic<bool> m_running = true;
    };
} // namespace voco::detail
