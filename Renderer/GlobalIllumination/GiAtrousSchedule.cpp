#include "Renderer/GlobalIllumination/GiAtrousSchedule.h"

#include <algorithm>

#include "Utilities/Assert.h"

namespace mukki
{

std::vector<GiAtrousStep> BuildGiAtrousSchedule(std::uint32_t iterations, std::uint32_t maxIterations)
{
    MRE_ASSERT(maxIterations > 0, "BuildGiAtrousSchedule requires at least one iteration slot");

    const std::uint32_t count = std::clamp(iterations, 1u, maxIterations);
    std::vector<GiAtrousStep> steps(count);

    GiAtrousTexture input = GiAtrousTexture::History;
    GiAtrousTexture output =
        count % 2u == 0u ? GiAtrousTexture::Noisy : GiAtrousTexture::Filtered;

    for (std::uint32_t index = 0; index < count; ++index)
    {
        steps[index] = GiAtrousStep{input, output};
        input = output;
        output = output == GiAtrousTexture::Noisy ? GiAtrousTexture::Filtered : GiAtrousTexture::Noisy;
    }

    return steps;
}

} // namespace mukki
