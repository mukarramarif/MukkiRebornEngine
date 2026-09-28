#include "Renderer/Lighting/Light.h"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

namespace mukki
{

GpuLight ToGpuLight(const Light& light)
{
    const float directionLength = glm::length(light.direction);
    const glm::vec3 direction = directionLength > 0.0f ? light.direction / directionLength
                                                       : glm::vec3{0.0f, -1.0f, 0.0f};

    const float outerCone = std::clamp(light.outerConeAngle, 0.0f, 1.5533431f);
    const float innerCone = std::clamp(light.innerConeAngle, 0.0f, outerCone);

    GpuLight gpu{};
    gpu.positionRange = glm::vec4{light.position, std::max(light.range, 0.0f)};
    gpu.directionIntensity = glm::vec4{direction, light.intensity};
    gpu.color = glm::vec4{light.color, 1.0f};
    gpu.params = glm::vec4{
        static_cast<float>(light.type),
        std::cos(innerCone),
        std::cos(outerCone),
        0.0f};
    return gpu;
}

} // namespace mukki
