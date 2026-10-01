#include "Renderer/Deferred/GBuffer.h"

#include "Renderer/RHI/Device.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

namespace mukki
{

bool GBuffer::Init(rhi::Device& device, std::uint32_t width, std::uint32_t height)
{
    MRE_ASSERT(width > 0 && height > 0, "GBuffer dimensions must be greater than zero");
    MRE_ASSERT(m_albedo == nullptr, "GBuffer::Init called twice");

    m_width = width;
    m_height = height;
    return CreateTextures(device);
}

void GBuffer::Shutdown()
{
    m_specularF0.reset();
    m_motion.reset();
    m_depth.reset();
    m_position.reset();
    m_normal.reset();
    m_albedo.reset();
    m_width = 0;
    m_height = 0;
}

bool GBuffer::Resize(rhi::Device& device, std::uint32_t width, std::uint32_t height)
{
    if (width == 0 || height == 0)
    {
        return false;
    }

    if (width == m_width && height == m_height && m_albedo != nullptr)
    {
        return true;
    }

    MRE_INFO("Resizing the G-buffer to {}x{}", width, height);

    m_specularF0.reset();
    m_motion.reset();
    m_depth.reset();
    m_position.reset();
    m_normal.reset();
    m_albedo.reset();

    m_width = width;
    m_height = height;
    return CreateTextures(device);
}

bool GBuffer::CreateTextures(rhi::Device& device)
{
    const rhi::TextureDesc albedoDesc{
        m_width,
        m_height,
        rhi::Format::R8G8B8A8Unorm,
        rhi::TextureUsage::ColorAttachment};
    const rhi::TextureDesc normalDesc{
        m_width,
        m_height,
        rhi::Format::R8G8B8A8Unorm,
        rhi::TextureUsage::ColorAttachment};
    const rhi::TextureDesc positionDesc{
        m_width,
        m_height,
        rhi::Format::R16G16B16A16Sfloat,
        rhi::TextureUsage::ColorAttachment};
    const rhi::TextureDesc depthDesc{
        m_width,
        m_height,
        rhi::Format::D32Sfloat,
        rhi::TextureUsage::DepthAttachment};
    const rhi::TextureDesc motionDesc{
        m_width,
        m_height,
        rhi::Format::R16G16Sfloat,
        rhi::TextureUsage::ColorAttachment};
    const rhi::TextureDesc specularF0Desc{
        m_width,
        m_height,
        rhi::Format::R8G8B8A8Unorm,
        rhi::TextureUsage::ColorAttachment};

    m_albedo = device.CreateTexture(albedoDesc);
    m_normal = device.CreateTexture(normalDesc);
    m_position = device.CreateTexture(positionDesc);
    m_depth = device.CreateTexture(depthDesc);
    m_motion = device.CreateTexture(motionDesc);
    m_specularF0 = device.CreateTexture(specularF0Desc);

    if (m_albedo == nullptr || m_normal == nullptr || m_position == nullptr || m_depth == nullptr ||
        m_motion == nullptr || m_specularF0 == nullptr)
    {
        MRE_CRITICAL("Failed to create the G-buffer textures");
        Shutdown();
        return false;
    }

    return true;
}

} // namespace mukki
