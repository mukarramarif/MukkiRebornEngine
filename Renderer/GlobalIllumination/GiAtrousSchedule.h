#pragma once

#include <cstdint>
#include <vector>

namespace mukki
{

enum class GiAtrousTexture : std::uint32_t
{
    History = 0,
    Noisy = 1,
    Filtered = 2,
};

struct GiAtrousStep
{
    GiAtrousTexture input = GiAtrousTexture::History;
    GiAtrousTexture output = GiAtrousTexture::Noisy;
};

// Builds the ping-pong schedule of the a-trous denoiser: the first iteration reads the reprojected
// history texture and every later iteration reads the previous output. The last iteration always
// writes the filtered texture so the upsample pass has a fixed source. Iterations are clamped to
// [1, maxIterations].
[[nodiscard]] std::vector<GiAtrousStep> BuildGiAtrousSchedule(
    std::uint32_t iterations,
    std::uint32_t maxIterations);

} // namespace mukki
