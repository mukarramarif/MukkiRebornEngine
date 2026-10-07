#include "Renderer/BufferManager.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"

namespace mukki
{

BufferManager::BufferManager(rhi::Device& device) : m_device(&device) {}

rhi::Buffer& BufferManager::CreateVertexBuffer(const void* data, std::uint64_t size)
{
    MRE_ASSERT(data != nullptr, "BufferManager::CreateVertexBuffer requires vertex data");
    const bool deviceAddress = m_device->IsRayTracingSupported();
    return Store(m_device->CreateBuffer(
        rhi::BufferDesc{size, rhi::BufferUsage::Vertex, false, deviceAddress},
        data,
        size));
}

rhi::Buffer& BufferManager::CreateIndexBuffer(const void* data, std::uint64_t size)
{
    MRE_ASSERT(data != nullptr, "BufferManager::CreateIndexBuffer requires index data");
    const bool deviceAddress = m_device->IsRayTracingSupported();
    return Store(m_device->CreateBuffer(
        rhi::BufferDesc{size, rhi::BufferUsage::Index, false, deviceAddress},
        data,
        size));
}

rhi::Buffer& BufferManager::CreateUniformBuffer(std::uint64_t size)
{
    MRE_ASSERT(size > 0, "BufferManager::CreateUniformBuffer requires a positive size");
    return Store(m_device->CreateBuffer(rhi::BufferDesc{size, rhi::BufferUsage::Uniform, true}, nullptr, 0));
}

void BufferManager::Clear()
{
    m_buffers.clear();
}

rhi::Buffer& BufferManager::Store(std::unique_ptr<rhi::Buffer> buffer)
{
    MRE_ASSERT(buffer != nullptr, "Buffer creation failed");
    m_buffers.push_back(std::move(buffer));
    return *m_buffers.back();
}

} // namespace mukki
