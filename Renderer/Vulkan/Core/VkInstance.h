#pragma once

#include <string>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace mukki::vulkan
{

class VkInstance
{
public:
    VkInstance() = default;
    ~VkInstance() = default;

    VkInstance(const VkInstance&) = delete;
    VkInstance& operator=(const VkInstance&) = delete;

    bool Init(const std::string& applicationName, void* nativeWindowHandle, bool enableValidation);
    void Shutdown();

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
    [[nodiscard]] bool IsValidationEnabled() const { return m_validationEnabled; }
    [[nodiscard]] const vk::raii::Instance& GetInstance() const { return m_instance; }
    [[nodiscard]] const vk::raii::SurfaceKHR& GetSurface() const { return m_surface; }

private:
    bool CreateInstance(const std::string& applicationName, bool enableValidation);
    bool CreateSurface(void* nativeWindowHandle);

    vk::raii::Context m_context{};
    vk::raii::Instance m_instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT m_debugMessenger{nullptr};
    vk::raii::SurfaceKHR m_surface{nullptr};
    bool m_validationEnabled = false;
    bool m_initialized = false;
};

} // namespace mukki::vulkan
