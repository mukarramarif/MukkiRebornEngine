#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Renderer/RHI/Device.h"
#include "Renderer/RHI/Resources.h"

namespace mukki
{

class BufferManager
{
public:
    explicit BufferManager(rhi::Device& device);
    ~BufferManager() = default;

    BufferManager(const BufferManager&) = delete;
    BufferManager& operator=(const BufferManager&) = delete;

    [[nodiscard]] rhi::Buffer& CreateVertexBuffer(const void* data, std::uint64_t size);
    [[nodiscard]] rhi::Buffer& CreateIndexBuffer(const void* data, std::uint64_t size);
    [[nodiscard]] rhi::Buffer& CreateUniformBuffer(std::uint64_t size);
    void Clear();

private:
    [[nodiscard]] rhi::Buffer& Store(std::unique_ptr<rhi::Buffer> buffer);

    rhi::Device* m_device = nullptr;
    std::vector<std::unique_ptr<rhi::Buffer>> m_buffers{};
};

} // namespace mukki
