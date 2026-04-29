#pragma once
#include <vulkan/vulkan.h>
#include <memory>
#include "types.h"
#include "buffer.h"
#include "pipeline.h"

namespace voco
{
    class CommandList
    {
    public:
        ~CommandList();

        CommandList(const CommandList&) = delete;
        CommandList& operator=(const CommandList&) = delete;

        CommandList(CommandList&&) noexcept;
        CommandList& operator=(CommandList&&) noexcept;

        void bindPipeline(const ComputePipeline& pipeline);
        void bindBuffer(uint32_t set, uint32_t binding, Buffer& buffer, Access access = Access::ReadWrite);

        template<typename T>
        void setPushConstants(const T& data)
        {
            setPushConstantsImpl(&data, sizeof(T));
        }

        void dispatch(uint32_t x, uint32_t y, uint32_t z);

    private:
        friend class Device;

        CommandList() = default;

        void setPushConstantsImpl(const void* data, uint32_t size);

        struct State;
        std::unique_ptr<State> m_state;
    };

} // namespace voco
