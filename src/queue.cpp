#include "queue.h"
#include "utils.h"
#include <vector>

namespace voco::detail
{
    Queue::Queue(VkDevice device, VkQueue queue, uint32_t queueFamilyIndex)
        : m_device(device)
        , m_queue(queue)
        , m_queueFamilyIndex(queueFamilyIndex)
    {
        VkSemaphoreTypeCreateInfo semaphoreTypeInfo{};
        semaphoreTypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        semaphoreTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        semaphoreTypeInfo.initialValue = 0;

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreInfo.pNext = &semaphoreTypeInfo;

        VK_CHECK(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_timelineSemaphore));
    }

    Queue::~Queue()
    {
        for (auto& cb : m_inFlight)
            vkDestroyCommandPool(m_device, cb.pool, nullptr);
        for (auto& cb : m_pool)
            vkDestroyCommandPool(m_device, cb.pool, nullptr);

        vkDestroySemaphore(m_device, m_timelineSemaphore, nullptr);
    }

    TrackedCommandBuffer Queue::acquire()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_pool.empty())
            {
                TrackedCommandBuffer cb = std::move(m_pool.back());
                m_pool.pop_back();
                return cb;
            }
        }

        TrackedCommandBuffer cb;

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = m_queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &cb.pool));

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cb.pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &cb.cmd));

        return cb;
    }

    uint64_t Queue::submit(TrackedCommandBuffer cb)
    {
        retire();

        cb.submissionID = ++m_lastSubmittedID;

        VkCommandBufferSubmitInfo cmdInfo{};
        cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmdInfo.commandBuffer = cb.cmd;

        std::vector<VkSemaphoreSubmitInfo> signalInfos;
        signalInfos.reserve(1 + m_signalSemaphores.size());

        VkSemaphoreSubmitInfo timelineSignal{};
        timelineSignal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        timelineSignal.semaphore = m_timelineSemaphore;
        timelineSignal.value = m_lastSubmittedID;
        timelineSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        signalInfos.push_back(timelineSignal);

        for (size_t i = 0; i < m_signalSemaphores.size(); ++i)
        {
            VkSemaphoreSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            info.semaphore = m_signalSemaphores[i];
            info.value = m_signalSemaphoreValues[i];
            info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            signalInfos.push_back(info);
        }

        std::vector<VkSemaphoreSubmitInfo> waitInfos;
        waitInfos.reserve(m_waitSemaphores.size());

        for (size_t i = 0; i < m_waitSemaphores.size(); ++i)
        {
            VkSemaphoreSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            info.semaphore = m_waitSemaphores[i];
            info.value = m_waitSemaphoreValues[i];
            info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            waitInfos.push_back(info);
        }

        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdInfo;
        submitInfo.signalSemaphoreInfoCount = static_cast<uint32_t>(signalInfos.size());
        submitInfo.pSignalSemaphoreInfos = signalInfos.data();
        submitInfo.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size());
        submitInfo.pWaitSemaphoreInfos = waitInfos.data();

        VK_CHECK(vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE));

        m_waitSemaphores.clear();
        m_waitSemaphoreValues.clear();
        m_signalSemaphores.clear();
        m_signalSemaphoreValues.clear();

        m_inFlight.push_back(std::move(cb));

        return m_lastSubmittedID;
    }

    void Queue::retire()
    {
        uint64_t completed = getLastFinishedID();

        auto it = m_inFlight.begin();
        while (it != m_inFlight.end())
        {
            if (it->submissionID <= completed)
            {
                vkResetCommandPool(m_device, it->pool, 0);
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_pool.push_back(std::move(*it));
                }
                it = m_inFlight.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void Queue::wait(uint64_t value)
    {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.flags = 0;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &m_timelineSemaphore;
        waitInfo.pValues = &value;

        vkWaitSemaphores(m_device, &waitInfo, std::numeric_limits<uint64_t>::max());
    }

    uint64_t Queue::getLastFinishedID()
    {
        vkGetSemaphoreCounterValue(m_device, m_timelineSemaphore, &m_lastFinishedID);
        return m_lastFinishedID;
    }

    void Queue::addWaitSemaphore(VkSemaphore semaphore, uint64_t value)
    {
        m_waitSemaphores.push_back(semaphore);
        m_waitSemaphoreValues.push_back(value);
    }

    void Queue::addSignalSemaphore(VkSemaphore semaphore, uint64_t value)
    {
        m_signalSemaphores.push_back(semaphore);
        m_signalSemaphoreValues.push_back(value);
    }
}
