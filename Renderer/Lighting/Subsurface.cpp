#include "Renderer/Lighting/Subsurface.h"

#include <algorithm>
#include <cmath>

namespace mukki
{
namespace
{

// Per-channel scatter radius ratios: red penetrates deepest (skin-like), blue the least.
constexpr float kChannelRadiusScale[3] = {1.0f, 0.7f, 0.5f};

// Gaussian sigma relative to the scattering radius: ~99% of the energy falls inside the radius.
constexpr float kSigmaScale = 0.4f;

} // namespace

void BuildSubsurfaceLut(std::vector<float>& pixels)
{
    pixels.assign(
        static_cast<std::size_t>(kSubsurfaceLutWidth) * kSubsurfaceLutHeight * 4,
        0.0f);

    const float logMin = std::log(kSubsurfaceLutRadiusMinPx);
    const float logMax = std::log(kSubsurfaceLutRadiusMaxPx);
    const float logRange = std::max(logMax - logMin, 1.0e-4f);

    for (std::uint32_t y = 0; y < kSubsurfaceLutHeight; ++y)
    {
        const float bucket = kSubsurfaceLutHeight > 1
                                 ? static_cast<float>(y) / static_cast<float>(kSubsurfaceLutHeight - 1)
                                 : 0.0f;
        const float radiusPx = std::exp(logMin + logRange * bucket);

        for (std::uint32_t x = 0; x < kSubsurfaceLutWidth; ++x)
        {
            const float u = kSubsurfaceLutWidth > 1
                                ? static_cast<float>(x) / static_cast<float>(kSubsurfaceLutWidth - 1)
                                : 0.0f;
            const float distance = u * radiusPx;
            const std::size_t offset =
                (static_cast<std::size_t>(y) * kSubsurfaceLutWidth + x) * 4;

            for (int channel = 0; channel < 3; ++channel)
            {
                const float sigma =
                    std::max(radiusPx * kChannelRadiusScale[channel] * kSigmaScale, 1.0e-4f);
                const float normalized = distance / sigma;
                pixels[offset + static_cast<std::size_t>(channel)] =
                    std::exp(-0.5f * normalized * normalized);
            }
            pixels[offset + 3] = 1.0f;
        }
    }
}

} // namespace mukki
