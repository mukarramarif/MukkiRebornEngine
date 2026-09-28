#include "Renderer/Lighting/ShadowCascade.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/matrix.hpp>
#include <glm/vec2.hpp>

namespace mukki
{

std::vector<ShadowCascade> ComputeShadowCascades(
    const FrameView& view,
    const glm::vec3& lightDirection,
    const ShadowSettings& settings)
{
    const std::uint32_t cascadeCount = std::clamp(settings.cascadeCount, 1u, kShadowCascadeCount);
    std::vector<ShadowCascade> cascades(cascadeCount);

    const float nearPlane = std::max(view.nearPlane, 0.01f);
    const float farPlane = std::max(settings.distance, nearPlane + 1.0f);
    const float resolution = static_cast<float>(std::max(settings.resolution, 1u));

    const float directionLength = glm::length(lightDirection);
    const glm::vec3 direction = directionLength > 1e-4f ? lightDirection / directionLength
                                                        : glm::vec3{0.0f, -1.0f, 0.0f};
    const glm::vec3 up = std::abs(direction.y) > 0.99f ? glm::vec3{0.0f, 0.0f, 1.0f}
                                                       : glm::vec3{0.0f, 1.0f, 0.0f};

    const glm::mat4 inverseViewProjection = glm::inverse(view.viewProjection);

    const auto ndcDepthAt = [&view](float depth) {
        const glm::vec4 clip = view.projection * glm::vec4{0.0f, 0.0f, -depth, 1.0f};
        return clip.z / clip.w;
    };

    const auto worldPoint = [&](float ndcX, float ndcY, float depth) {
        const glm::vec4 world =
            inverseViewProjection * glm::vec4{ndcX, ndcY, ndcDepthAt(depth), 1.0f};
        return glm::vec3{world} / world.w;
    };

    constexpr float kCornerNdc[4][2] = {{-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, 1.0f}};

    float splitNear = nearPlane;
    for (std::uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex)
    {
        const float blend = static_cast<float>(cascadeIndex + 1) / static_cast<float>(cascadeCount);
        const float logSplit = nearPlane * std::pow(farPlane / nearPlane, blend);
        const float uniformSplit = nearPlane + (farPlane - nearPlane) * blend;
        const float splitFar = settings.lambda * logSplit + (1.0f - settings.lambda) * uniformSplit;

        glm::vec3 corners[8];
        for (std::uint32_t corner = 0; corner < 4; ++corner)
        {
            corners[corner] = worldPoint(kCornerNdc[corner][0], kCornerNdc[corner][1], splitNear);
            corners[corner + 4] = worldPoint(kCornerNdc[corner][0], kCornerNdc[corner][1], splitFar);
        }

        glm::vec3 center{0.0f};
        for (const glm::vec3& corner : corners)
        {
            center += corner;
        }
        center /= 8.0f;

        float radius = 0.0f;
        for (const glm::vec3& corner : corners)
        {
            radius = std::max(radius, glm::length(corner - center));
        }
        radius = std::max(radius, 1e-3f);

        const float padding = radius;
        const glm::mat4 lightView = glm::lookAt(center - direction * (radius + padding), center, up);

        // Snap the projection center to the texel grid so the map stays aligned to world space
        // while the camera moves.
        glm::vec2 minXY{std::numeric_limits<float>::max()};
        glm::vec2 maxXY{std::numeric_limits<float>::lowest()};
        for (const glm::vec3& corner : corners)
        {
            const glm::vec3 lightSpace = glm::vec3{lightView * glm::vec4{corner, 1.0f}};
            minXY = glm::min(minXY, glm::vec2{lightSpace.x, lightSpace.y});
            maxXY = glm::max(maxXY, glm::vec2{lightSpace.x, lightSpace.y});
        }

        // Use the bounding sphere radius as the ortho half extent: it only depends on the camera
        // projection and the split range, so the shadow scale stays constant while the camera
        // translates or rotates and the shadows do not swim.
        const float halfExtent = radius;
        const glm::vec2 centerXY = (minXY + maxXY) * 0.5f;
        const float texelSize = 2.0f * halfExtent / resolution;
        const glm::vec2 snappedCenter = texelSize > 0.0f
            ? glm::floor(centerXY / texelSize) * texelSize
            : centerXY;

        const float lightNear = 0.01f;
        const float lightFar = 2.0f * radius + 2.0f * padding;
        const glm::mat4 lightProjection = glm::ortho(
            snappedCenter.x - halfExtent,
            snappedCenter.x + halfExtent,
            snappedCenter.y - halfExtent,
            snappedCenter.y + halfExtent,
            lightNear,
            lightFar);

        cascades[cascadeIndex].lightViewProjection = lightProjection * lightView;
        cascades[cascadeIndex].splitNear = splitNear;
        cascades[cascadeIndex].splitFar = splitFar;

        splitNear = splitFar;
    }

    return cascades;
}

} // namespace mukki
