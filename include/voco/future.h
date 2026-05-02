#pragma once
#include <cstdint>
#include <functional>

namespace voco
{
    namespace detail { class Queue; }

    class Future
    {
    public:
        Future() = default;
        ~Future();

        Future(const Future&) = delete;
        Future& operator=(const Future&) = delete;

        Future(Future&&) noexcept;
        Future& operator=(Future&&) noexcept;

        void wait();
        bool valid() const { return m_queue != nullptr; }

    private:
        friend class Device;

        Future(detail::Queue* queue, uint64_t submissionID);
        Future(detail::Queue* queue, uint64_t submissionID, std::function<void()> onComplete);

        detail::Queue* m_queue = nullptr;
        uint64_t m_submissionID = 0;
        std::function<void()> m_onComplete;
    };
} // namespace voco
