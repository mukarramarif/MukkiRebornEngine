#pragma once

#include <cstdint>

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

    virtual void BeginSwapchainRendering(const ClearColor& clearColor, bool clear = true) = 0;
    virtual void EndSwapchainRendering() = 0;

    virtual void BindPipeline(const Pipeline& pipeline) = 0;
    virtual void BindVertexBuffer(const Buffer& buffer) = 0;
    virtual void SetPushConstants(const void* data, std::uint32_t size, ShaderStageMask stages) = 0;
    virtual void Draw(std::uint32_t vertexCount, std::uint32_t firstVertex = 0) = 0;

protected:
    CommandList() = default;
};

} // namespace mukki::rhi
