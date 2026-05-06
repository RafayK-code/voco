#include "retirement_queue.h"

namespace voco::detail
{
    RetirementQueue::RetirementQueue(Queue& queue)
        : m_queue(queue)
    {
        m_retirementThread = std::thread(&RetirementQueue::retirementLoop, this);
    }

    RetirementQueue::~RetirementQueue()
    {
        m_running = false;
        m_retirementCV.notify_one();
        if (m_retirementThread.joinable())
            m_retirementThread.join();

        m_queue.wait(m_queue.getLastSubmittedID());
        for (auto& p : m_pending)
            p.destroy();

        m_pending.clear();
    }

    void RetirementQueue::push(PendingDestroy pending)
    {
        {
            std::lock_guard<std::mutex> lock(m_retirementMutex);

            auto it = std::upper_bound(m_pending.begin(), m_pending.end(),pending.submissionID, [](uint64_t id, const PendingDestroy& e) {
                return id < e.submissionID;
            });

            m_pending.insert(it, std::move(pending));
        }
        m_retirementCV.notify_one();
    }

    void RetirementQueue::retirementLoop()
    {
        while (m_running)
        {
            std::unique_lock<std::mutex> lock(m_retirementMutex);
            m_retirementCV.wait(lock, [this]{ return !m_pending.empty() || !m_running; });

            if (m_pending.empty())
                continue;

            std::vector<PendingDestroy> deletedEntries;
            uint64_t lastFinishedID = m_queue.getLastFinishedID();

            while (!m_pending.empty() && m_pending.front().submissionID <= lastFinishedID)
            {
                deletedEntries.push_back(std::move(m_pending.front()));
                m_pending.pop_front();
            }

            lock.unlock();

            for (auto& entry : deletedEntries)
                entry.destroy();
        }
    }
}