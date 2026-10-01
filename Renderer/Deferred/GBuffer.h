#pragma once

#include <cstdint>
#include <memory>

#include "Renderer/RHI/Resources.h"

namespace mukki::rhi
{
class Device;
}

namespace mukki
{

class GBuffer
{
public:
    GBuffer() = default;
    ~GBuffer() = default;

    GBuffer(const GBuffer&) = delete;
    GBuffer& operator=(const GBuffer&) = delete;

    bool Init(rhi::Device& device, std::uint32_t width, std::uint32_t height);
    void Shutdown();
    bool Resize(rhi::Device& device, std::uint32_t width, std::uint32_t height);

    [[nodiscard]] std::uint32_t GetWidth() const { return m_width; }
    [[nodiscard]] std::uint32_t GetHeight() const { return m_height; }
    [[nodiscard]] bool IsInitialized() const { return m_albedo != nullptr; }

    [[nodiscard]] rhi::Texture& GetAlbedo() { return *m_albedo; }
    [[nodiscard]] rhi::Texture& GetNormal() { return *m_normal; }
    [[nodiscard]] rhi::Texture& GetPosition() { return *m_position; }
    [[nodiscard]] rhi::Texture& GetDepth() { return *m_depth; }
    [[nodiscard]] rhi::Texture& GetMotion() { return *m_motion; }

private:
    bool CreateTextures(rhi::Device& device);

    std::unique_ptr<rhi::Texture> m_albedo{};
    std::unique_ptr<rhi::Texture> m_normal{};
    std::unique_ptr<rhi::Texture> m_position{};
    std::unique_ptr<rhi::Texture> m_depth{};
    std::unique_ptr<rhi::Texture> m_motion{};
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
};

} // namespace mukki
