#pragma once

#include "Renderer/RHI/Device.h"
#include "Renderer/RHI/Resources.h"

#include <imgui.h>

struct GLFWwindow;

namespace mukki::vulkan
{

class VkDevice;

class UiLayer
{
public:
    UiLayer() = default;
    ~UiLayer();

    UiLayer(const UiLayer&) = delete;
    UiLayer& operator=(const UiLayer&) = delete;

    bool Init(rhi::Device& device, GLFWwindow* window);
    void Shutdown();

    void BeginFrame();
    void EndFrame(rhi::CommandList& commandList);

    [[nodiscard]] ImTextureID AddTexture(const rhi::Texture& texture);
    void RemoveTexture(ImTextureID textureId);

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }

private:
    VkDevice* m_device = nullptr;
    bool m_initialized = false;
};

} // namespace mukki::vulkan
