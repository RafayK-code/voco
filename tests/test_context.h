#pragma once
#include <voco/voco.h>
#include <memory>

class TestContext
{
public:
    TestContext();
    ~TestContext();

    voco::Device& device() { return *m_device; }

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkDevice m_vkDevice = VK_NULL_HANDLE;
    std::unique_ptr<voco::Device> m_device;
};

extern TestContext* g_ctx;
