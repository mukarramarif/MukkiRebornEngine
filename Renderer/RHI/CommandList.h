#pragma once

#include "Renderer/RHI/Resources.h"

namespace mukki::rhi
{

class CommandList
{
public:
    virtual ~CommandList() = default;

    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;

    virtual void Begin() = 0;
    virtual void End() = 0;

    virtual void BeginSwapchainRendering(const ClearColor& clearColor) = 0;
    virtual void EndSwapchainRendering() = 0;

protected:
    CommandList() = default;
};

} // namespace mukki::rhi
