#include "Renderer/Vulkan/UiLayer.h"

#include "Application/UI/UiStyles.h"
#include "Renderer/Vulkan/Core/VkCommandList.h"
#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Core/VkDevice.h"
#include "Renderer/Vulkan/Resources/Texture.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace mukki::vulkan
{

UiLayer::~UiLayer()
{
    Shutdown();
}

bool UiLayer::Init(rhi::Device& device, GLFWwindow* window)
{
    MRE_ASSERT(!m_initialized, "UiLayer::Init called twice");
    MRE_ASSERT(window != nullptr, "UiLayer::Init requires a valid GLFW window");

    auto* vkDevice = dynamic_cast<VkDevice*>(&device);
    MRE_ASSERT(vkDevice != nullptr, "UiLayer requires a Vulkan device");

    m_device = vkDevice;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#ifdef IMGUI_HAS_DOCK
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#endif

    mukki::ui::ApplyDarkTheme();

    if (!ImGui_ImplGlfw_InitForVulkan(window, true))
    {
        MRE_CRITICAL("Failed to initialize the ImGui GLFW backend");
        ImGui::DestroyContext();
        m_device = nullptr;
        return false;
    }

    const rhi::SwapchainInfo& swapchainInfo = vkDevice->GetSwapchainInfo();
    const VkFormat colorFormat = static_cast<VkFormat>(ToVkFormat(swapchainInfo.format));

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = *vkDevice->GetVkInstance().GetInstance();
    initInfo.PhysicalDevice = *vkDevice->GetPhysicalDevice();
    initInfo.Device = *vkDevice->GetHandle();
    initInfo.QueueFamily = vkDevice->GetGraphicsQueueFamily();
    initInfo.Queue = *vkDevice->GetGraphicsQueue();
    initInfo.DescriptorPoolSize = 64;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = swapchainInfo.imageCount;
    initInfo.MinAllocationSize = 1024 * 1024;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;

    if (!ImGui_ImplVulkan_Init(&initInfo))
    {
        MRE_CRITICAL("Failed to initialize the ImGui Vulkan backend");
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_device = nullptr;
        return false;
    }

    m_initialized = true;

#ifdef IMGUI_HAS_DOCK
    constexpr const char* kDockingStatus = "on";
#else
    constexpr const char* kDockingStatus = "off";
#endif
    MRE_INFO("ImGui layer initialized (docking {})", kDockingStatus);
    return true;
}

void UiLayer::Shutdown()
{
    if (m_initialized)
    {
        if (m_device != nullptr)
        {
            m_device->WaitIdle();
        }

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_initialized = false;
    }

    m_device = nullptr;
}

void UiLayer::BeginFrame()
{
    MRE_ASSERT(m_initialized, "UiLayer::BeginFrame called before UiLayer::Init");

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UiLayer::EndFrame(rhi::CommandList& commandList)
{
    MRE_ASSERT(m_initialized, "UiLayer::EndFrame called before UiLayer::Init");

    auto* vkCommandList = dynamic_cast<VkCommandList*>(&commandList);
    MRE_ASSERT(vkCommandList != nullptr, "UiLayer requires a Vulkan command list");

    ImGui::Render();

    ImDrawData* drawData = ImGui::GetDrawData();
    if (drawData != nullptr && drawData->CmdListsCount > 0)
    {
        ImGui_ImplVulkan_RenderDrawData(drawData, *vkCommandList->GetHandle());
    }
}

ImTextureID UiLayer::AddTexture(const rhi::Texture& texture)
{
    MRE_ASSERT(m_initialized, "UiLayer::AddTexture called before UiLayer::Init");

    const auto* vkTexture = dynamic_cast<const VkTexture*>(&texture);
    MRE_ASSERT(vkTexture != nullptr, "UiLayer requires Vulkan textures");
    MRE_ASSERT(
        vkTexture->GetLayout() == vk::ImageLayout::eShaderReadOnlyOptimal,
        "ImGui textures must be in a shader readable layout before registration");

    const ::VkDescriptorSet descriptorSet =
        ImGui_ImplVulkan_AddTexture(vkTexture->GetImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return reinterpret_cast<ImTextureID>(descriptorSet);
}

void UiLayer::RemoveTexture(ImTextureID textureId)
{
    if (textureId == 0)
    {
        return;
    }

    ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<::VkDescriptorSet>(textureId));
}

} // namespace mukki::vulkan
