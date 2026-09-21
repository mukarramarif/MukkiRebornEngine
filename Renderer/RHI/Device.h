#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Renderer/RHI/CommandList.h"
#include "Renderer/RHI/Resources.h"

namespace mukki::rhi
{

struct DeviceDesc
{
    void* nativeWindowHandle = nullptr;
    std::string applicationName = "MukkiRebornEngine";
    Extent2D initialExtent{1280, 720};
    bool enableValidation = true;
    std::uint32_t framesInFlight = 2;
};

struct SwapchainInfo
{
    Format format = Format::Undefined;
    Extent2D extent{};
    std::uint32_t imageCount = 0;
};

class Device
{
public:
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    virtual bool BeginFrame() = 0;
    virtual void EndFrame() = 0;
    virtual void WaitIdle() = 0;
    virtual void Resize(const Extent2D& extent) = 0;

    [[nodiscard]] virtual CommandList& GetCommandList() = 0;
    [[nodiscard]] virtual const SwapchainInfo& GetSwapchainInfo() const = 0;
    [[nodiscard]] virtual std::uint32_t GetFrameIndex() const = 0;
    [[nodiscard]] virtual std::uint32_t GetFramesInFlight() const = 0;

    [[nodiscard]] virtual std::unique_ptr<Buffer> CreateBuffer(
        const BufferDesc& desc,
        const void* initialData = nullptr,
        std::uint64_t initialDataSize = 0) = 0;
    [[nodiscard]] virtual std::unique_ptr<Texture> CreateTexture(const TextureDesc& desc) = 0;
    [[nodiscard]] virtual std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) = 0;
    [[nodiscard]] virtual std::unique_ptr<Shader> CreateShader(ShaderStage stage, const std::string& spirvPath) = 0;
    [[nodiscard]] virtual std::unique_ptr<Pipeline> CreatePipeline(const PipelineDesc& desc) = 0;
    [[nodiscard]] virtual std::unique_ptr<DescriptorSetLayout> CreateDescriptorSetLayout(
        const std::vector<DescriptorBinding>& bindings) = 0;
    [[nodiscard]] virtual std::unique_ptr<DescriptorSet> CreateDescriptorSet(const DescriptorSetLayout& layout) = 0;

protected:
    Device() = default;
};

} // namespace mukki::rhi
