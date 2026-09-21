#include "Renderer/Vulkan/Core/VkInstance.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <algorithm>
#include <string_view>
#include <vector>

#include <GLFW/glfw3.h>

namespace mukki::vulkan
{
namespace
{

constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    const vk::DebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData)
{
    (void)messageTypes;
    (void)userData;

    if (callbackData == nullptr)
    {
        return vk::False;
    }

    if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
    {
        MRE_ERROR("[vulkan] {}", callbackData->pMessage);
    }
    else if (messageSeverity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
    {
        MRE_WARN("[vulkan] {}", callbackData->pMessage);
    }
    else
    {
        MRE_DEBUG("[vulkan] {}", callbackData->pMessage);
    }

    return vk::False;
}

bool IsValidationLayerAvailable()
{
    const auto layers = vk::enumerateInstanceLayerProperties();
    return std::any_of(layers.begin(), layers.end(), [](const vk::LayerProperties& layer) {
        return std::string_view{layer.layerName} == kValidationLayerName;
    });
}

} // namespace

bool VkInstance::Init(const std::string& applicationName, void* nativeWindowHandle, bool enableValidation)
{
    MRE_ASSERT(!m_initialized, "VkInstance::Init called twice");
    MRE_ASSERT(nativeWindowHandle != nullptr, "VkInstance::Init requires a native window handle");

    if (!CreateInstance(applicationName, enableValidation))
    {
        return false;
    }

    if (!CreateSurface(nativeWindowHandle))
    {
        Shutdown();
        return false;
    }

    m_initialized = true;
    return true;
}

void VkInstance::Shutdown()
{
    m_surface = vk::raii::SurfaceKHR{nullptr};
    m_debugMessenger = vk::raii::DebugUtilsMessengerEXT{nullptr};
    m_instance = vk::raii::Instance{nullptr};
    m_validationEnabled = false;
    m_initialized = false;
}

bool VkInstance::CreateInstance(const std::string& applicationName, bool enableValidation)
{
    uint32_t extensionCount = 0;
    const char** requiredExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (requiredExtensions == nullptr)
    {
        MRE_CRITICAL("GLFW did not return the required Vulkan instance extensions");
        return false;
    }

    std::vector<const char*> extensions{requiredExtensions, requiredExtensions + extensionCount};

    m_validationEnabled = enableValidation && IsValidationLayerAvailable();
    if (enableValidation && !m_validationEnabled)
    {
        MRE_WARN("Vulkan validation layers were requested but '{}' is not available", kValidationLayerName);
    }

    std::vector<const char*> layers{};
    if (m_validationEnabled)
    {
        layers.push_back(kValidationLayerName);
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    const vk::ApplicationInfo applicationInfo{
        applicationName.c_str(),
        VK_MAKE_API_VERSION(0, 0, 1, 0),
        "MukkiRebornEngine",
        VK_MAKE_API_VERSION(0, 0, 1, 0),
        VK_API_VERSION_1_4,
    };

    const vk::InstanceCreateInfo createInfo = vk::InstanceCreateInfo{}
                                                  .setPApplicationInfo(&applicationInfo)
                                                  .setPEnabledLayerNames(layers)
                                                  .setPEnabledExtensionNames(extensions);

    try
    {
        m_instance = m_context.createInstance(createInfo);
    }
    catch (const vk::SystemError& error)
    {
        MRE_CRITICAL("Failed to create the Vulkan instance: {}", error.what());
        return false;
    }

    if (m_validationEnabled)
    {
        const vk::DebugUtilsMessengerCreateInfoEXT messengerInfo = vk::DebugUtilsMessengerCreateInfoEXT{}
                                                                       .setMessageSeverity(
                                                                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
                                                                       .setMessageType(
                                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
                                                                       .setPfnUserCallback(&DebugCallback);
        m_debugMessenger = m_instance.createDebugUtilsMessengerEXT(messengerInfo);
    }

    const std::uint32_t loaderVersion = vk::enumerateInstanceVersion();
    MRE_INFO(
        "Vulkan instance created (validation {}, loader {}.{}.{})",
        m_validationEnabled ? "on" : "off",
        VK_API_VERSION_MAJOR(loaderVersion),
        VK_API_VERSION_MINOR(loaderVersion),
        VK_API_VERSION_PATCH(loaderVersion));

    return true;
}

bool VkInstance::CreateSurface(void* nativeWindowHandle)
{
    auto* window = static_cast<GLFWwindow*>(nativeWindowHandle);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult result = glfwCreateWindowSurface(*m_instance, window, nullptr, &surface);
    if (result != VK_SUCCESS)
    {
        MRE_CRITICAL("Failed to create the window surface (VkResult {})", static_cast<int>(result));
        return false;
    }

    m_surface = vk::raii::SurfaceKHR{m_instance, surface};
    return true;
}

} // namespace mukki::vulkan
