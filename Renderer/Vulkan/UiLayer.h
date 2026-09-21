#pragma once

#include "Renderer/RHI/Device.h"

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

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }

private:
    VkDevice* m_device = nullptr;
    bool m_initialized = false;
};

} // namespace mukki::vulkan
