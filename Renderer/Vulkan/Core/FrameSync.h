#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace mukki::vulkan
{

class FrameSync
{
public:
    FrameSync() = default;
    ~FrameSync() = default;

    FrameSync(const FrameSync&) = delete;
    FrameSync& operator=(const FrameSync&) = delete;

    void Init(const vk::raii::Device& device, std::uint32_t framesInFlight);
    void Shutdown();

    [[nodiscard]] std::uint32_t GetFramesInFlight() const { return static_cast<std::uint32_t>(m_frameValues.size()); }
    [[nodiscard]] vk::Semaphore GetImageAvailable(std::uint32_t frame) const { return *m_imageAvailable[frame]; }
    [[nodiscard]] vk::Semaphore GetTimeline() const { return *m_timeline; }
    [[nodiscard]] std::uint64_t GetFrameValue(std::uint32_t frame) const { return m_frameValues[frame]; }
    void SetFrameValue(std::uint32_t frame, std::uint64_t value) { m_frameValues[frame] = value; }
    [[nodiscard]] std::uint64_t AdvanceTimelineValue() { return ++m_timelineValue; }

private:
    vk::raii::Semaphore m_timeline{nullptr};
    std::vector<vk::raii::Semaphore> m_imageAvailable{};
    std::vector<std::uint64_t> m_frameValues{};
    std::uint64_t m_timelineValue = 0;
};

} // namespace mukki::vulkan
