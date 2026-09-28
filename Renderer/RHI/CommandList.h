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
    std::uint32_t layer = 0;
};

struct RenderingDesc
{
    std::vector<ColorAttachment> colorAttachments{};
    DepthAttachment depthAttachment{};
    bool secondaryCommandBuffers = false;
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
    virtual void Dispatch(std::uint32_t groupCountX, std::uint32_t groupCountY, std::uint32_t groupCountZ) = 0;

    // Memory dependency between shader stages, used when compute-produced storage data is read by
    // a later stage within the same command buffer.
    virtual void ShaderMemoryBarrier(ShaderStageMask before, ShaderStageMask after) = 0;

    // Dynamic depth bias for the currently bound graphics pipeline. Pipelines must be created with
    // depthBiasEnable for the values to take effect.
    virtual void SetDepthBias(float constantFactor, float slopeFactor, float clamp = 0.0f) = 0;

    // GPU timestamp zone markers, used by profilers such as Tracy. Zone names must remain
    // valid for the duration of the recording scope and zones must be balanced pairs.
    virtual void BeginGpuZone(const char* name) = 0;
    virtual void EndGpuZone() = 0;

    // Secondary command buffers inherit the active rendering scope of the primary command list that
    // executes them. They can be recorded on worker threads and replayed with ExecuteCommands.
    [[nodiscard]] virtual bool IsSecondary() const = 0;
    virtual void BeginSecondary(const RenderingDesc& desc) = 0;
    virtual void ExecuteCommands(const std::vector<const CommandList*>& secondaryCommandLists) = 0;

protected:
    CommandList() = default;
};

} // namespace mukki::rhi
