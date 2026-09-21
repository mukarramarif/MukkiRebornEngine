#include "Renderer/Vulkan/Core/FrameSync.h"

#include "Utilities/Assert.h"

namespace mukki::vulkan
{

void FrameSync::Init(const vk::raii::Device& device, std::uint32_t framesInFlight)
{
    MRE_ASSERT(framesInFlight >= 1 && framesInFlight <= 3, "FrameSync supports between 1 and 3 frames in flight");

    const vk::SemaphoreTypeCreateInfo timelineType = vk::SemaphoreTypeCreateInfo{}
                                                         .setSemaphoreType(vk::SemaphoreType::eTimeline)
                                                         .setInitialValue(0);
    const vk::SemaphoreCreateInfo timelineCreateInfo = vk::SemaphoreCreateInfo{}.setPNext(&timelineType);
    m_timeline = vk::raii::Semaphore{device, timelineCreateInfo};

    m_imageAvailable.clear();
    m_frameValues.assign(framesInFlight, 0);

    m_imageAvailable.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_imageAvailable.emplace_back(device, vk::SemaphoreCreateInfo{});
    }

    m_timelineValue = 0;
}

void FrameSync::Shutdown()
{
    m_imageAvailable.clear();
    m_frameValues.clear();
    m_timeline = vk::raii::Semaphore{nullptr};
    m_timelineValue = 0;
}

} // namespace mukki::vulkan
