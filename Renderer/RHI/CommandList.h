#pragma once

#include <cstdint>
#include <vector>

#include "Renderer/RHI/Resources.h"

namespace mukki::rhi
{

struct ColorAttachment
{
    const Texture* texture = nullptr;
    ClearColor clearColor{};
    bool clear = true;
};

struct DepthAttachment
{
    const Texture* texture = nullptr;
    float clearDepth = 1.0f;
    bool clear = true;
};

struct RenderingDesc
{
    std::vector<ColorAttachment> colorAttachments{};
    DepthAttachment depthAttachment{};
};

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
    virtual void BeginRendering(const RenderingDesc& desc) = 0;
    virtual void EndRendering() = 0;

    virtual void BindPipeline(const Pipeline& pipeline) = 0;
    virtual void BindDescriptorSet(std::uint32_t setIndex, const DescriptorSet& descriptorSet) = 0;
    virtual void BindVertexBuffer(const Buffer& buffer) = 0;
    virtual void BindIndexBuffer(const Buffer& buffer, bool use32BitIndices = true) = 0;
    virtual void SetPushConstants(const void* data, std::uint32_t size, ShaderStageMask stages) = 0;
    virtual void Draw(std::uint32_t vertexCount, std::uint32_t firstVertex = 0) = 0;
    virtual void DrawIndexed(std::uint32_t indexCount, std::uint32_t firstIndex = 0) = 0;

protected:
    CommandList() = default;
};

} // namespace mukki::rhi
