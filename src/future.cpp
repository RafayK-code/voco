#include <voco/future.h>
#include "queue.h"

namespace voco
{
    Future::Future(detail::Queue* queue, uint64_t submissionID)
        : m_queue(queue)
        , m_submissionID(submissionID)
    {}

    Future::Future(detail::Queue* queue, uint64_t submissionID, std::function<void()> onComplete)
        : m_queue(queue)
        , m_submissionID(submissionID)
        , m_onComplete(std::move(onComplete))
    {}

    Future::~Future()
    {
        if (m_onComplete)
            wait();
    }

    Future::Future(Future&& other) noexcept
        : m_queue(other.m_queue)
        , m_submissionID(other.m_submissionID)
        , m_onComplete(std::move(other.m_onComplete))
    {
        other.m_queue = nullptr;
        other.m_submissionID = 0;
    }

    Future& Future::operator=(Future&& other) noexcept
    {
        m_queue = other.m_queue;
        m_submissionID = other.m_submissionID;
        m_onComplete = std::move(other.m_onComplete);
        other.m_queue = nullptr;
        other.m_submissionID = 0;
        return *this;
    }

    void Future::wait()
    {
        if (m_queue && m_submissionID > 0)
            m_queue->wait(m_submissionID);

        if (m_onComplete)
        {
            m_onComplete();
            m_onComplete = nullptr;
        }
    }
}
