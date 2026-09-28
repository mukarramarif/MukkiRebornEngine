#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Renderer/Lighting/ClusterGrid.h"

namespace mukki
{
namespace
{

struct TestCamera
{
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::vec3 position{0.0f};
    float fovY = 1.0471975512f;
    float aspect = 1280.0f / 720.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
};

TestCamera MakeCamera()
{
    TestCamera camera{};
    camera.position = glm::vec3{4.0f, 2.5f, 5.0f};
    camera.view = glm::lookAt(camera.position, glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    camera.projection = glm::perspective(camera.fovY, camera.aspect, camera.nearPlane, camera.farPlane);
    camera.projection[1][1] *= -1.0f;
    camera.viewProjection = camera.projection * camera.view;
    return camera;
}

struct Bounds
{
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
};

// Mirrors Renderer/Shaders/cluster_culling.comp.slang.
Bounds ShaderClusterBounds(
    const TestCamera& camera,
    std::uint32_t screenWidth,
    std::uint32_t screenHeight,
    std::uint32_t tileX,
    std::uint32_t tileY,
    std::uint32_t sliceZ)
{
    const glm::mat4 inverseView = glm::inverse(camera.view);
    const float tanHalfFov = std::tan(camera.fovY * 0.5f);

    const float tileSize = static_cast<float>(kClusterTileSize);
    const float pixelMinX = static_cast<float>(tileX) * tileSize;
    const float pixelMinY = static_cast<float>(tileY) * tileSize;
    const float pixelMaxX = std::min(pixelMinX + tileSize, static_cast<float>(screenWidth));
    const float pixelMaxY = std::min(pixelMinY + tileSize, static_cast<float>(screenHeight));

    const float screenX = static_cast<float>(screenWidth);
    const float screenY = static_cast<float>(screenHeight);
    const glm::vec2 ndcMin = glm::vec2{pixelMinX / screenX, pixelMinY / screenY} * 2.0f - 1.0f;
    const glm::vec2 ndcMax = glm::vec2{pixelMaxX / screenX, pixelMaxY / screenY} * 2.0f - 1.0f;

    const float sliceRatio = camera.farPlane / camera.nearPlane;
    const float sliceCount = static_cast<float>(kClusterZSliceCount);
    const float nearSlice = camera.nearPlane * std::pow(sliceRatio, static_cast<float>(sliceZ) / sliceCount);
    const float farSlice = camera.nearPlane * std::pow(sliceRatio, static_cast<float>(sliceZ + 1) / sliceCount);

    const glm::vec3 right = glm::normalize(glm::vec3{inverseView[0]});
    const glm::vec3 up = glm::normalize(glm::vec3{inverseView[1]});
    const glm::vec3 forward = -glm::normalize(glm::vec3{inverseView[2]});
    const glm::vec3 eye{inverseView[3]};

    const auto worldPoint = [&](const glm::vec2& ndc, float depth) {
        const float x = ndc.x * camera.aspect * tanHalfFov * depth;
        const float y = -ndc.y * tanHalfFov * depth;
        return eye + right * x + up * y + forward * depth;
    };

    const glm::vec3 corners[8] = {
        worldPoint({ndcMin.x, ndcMin.y}, nearSlice),
        worldPoint({ndcMax.x, ndcMin.y}, nearSlice),
        worldPoint({ndcMin.x, ndcMax.y}, nearSlice),
        worldPoint({ndcMax.x, ndcMax.y}, nearSlice),
        worldPoint({ndcMin.x, ndcMin.y}, farSlice),
        worldPoint({ndcMax.x, ndcMin.y}, farSlice),
        worldPoint({ndcMin.x, ndcMax.y}, farSlice),
        worldPoint({ndcMax.x, ndcMax.y}, farSlice),
    };

    Bounds bounds{corners[0], corners[0]};
    for (const glm::vec3& corner : corners)
    {
        bounds.min = glm::min(bounds.min, corner);
        bounds.max = glm::max(bounds.max, corner);
    }
    return bounds;
}

// Independent reference: unprojects the tile corners through the inverse view-projection at the
// same view depths.
Bounds ReferenceClusterBounds(
    const TestCamera& camera,
    std::uint32_t screenWidth,
    std::uint32_t screenHeight,
    std::uint32_t tileX,
    std::uint32_t tileY,
    std::uint32_t sliceZ)
{
    const glm::mat4 inverseViewProjection = glm::inverse(camera.viewProjection);

    const float screenX = static_cast<float>(screenWidth);
    const float screenY = static_cast<float>(screenHeight);
    const float tileSize = static_cast<float>(kClusterTileSize);
    const float pixelMinX = static_cast<float>(tileX) * tileSize;
    const float pixelMinY = static_cast<float>(tileY) * tileSize;
    const float pixelMaxX = std::min(pixelMinX + tileSize, screenX);
    const float pixelMaxY = std::min(pixelMinY + tileSize, screenY);

    const glm::vec2 ndcMin = glm::vec2{pixelMinX / screenX, pixelMinY / screenY} * 2.0f - 1.0f;
    const glm::vec2 ndcMax = glm::vec2{pixelMaxX / screenX, pixelMaxY / screenY} * 2.0f - 1.0f;

    const float sliceRatio = camera.farPlane / camera.nearPlane;
    const float sliceCount = static_cast<float>(kClusterZSliceCount);
    const float depths[2] = {
        camera.nearPlane * std::pow(sliceRatio, static_cast<float>(sliceZ) / sliceCount),
        camera.nearPlane * std::pow(sliceRatio, static_cast<float>(sliceZ + 1) / sliceCount),
    };

    // NDC depth depends only on the view depth for a perspective projection. The point is already
    // in view space, so only the projection is applied here.
    const auto ndcDepthAt = [&](float depth) {
        const glm::vec4 clip = camera.projection * glm::vec4{0.0f, 0.0f, -depth, 1.0f};
        return clip.z / clip.w;
    };

    const auto unproject = [&](const glm::vec2& ndc, float depth) {
        const glm::vec4 world = inverseViewProjection * glm::vec4{ndc.x, ndc.y, ndcDepthAt(depth), 1.0f};
        return glm::vec3{world} / world.w;
    };

    const glm::vec2 ndcCorners[2][2] = {
        {ndcMin, glm::vec2{ndcMax.x, ndcMin.y}},
        {glm::vec2{ndcMin.x, ndcMax.y}, ndcMax},
    };

    Bounds bounds{};
    bool first = true;
    for (const float depth : depths)
    {
        for (const auto& row : ndcCorners)
        {
            for (const glm::vec2& ndc : row)
            {
                const glm::vec3 point = unproject(ndc, depth);
                if (first)
                {
                    bounds.min = point;
                    bounds.max = point;
                    first = false;
                }
                else
                {
                    bounds.min = glm::min(bounds.min, point);
                    bounds.max = glm::max(bounds.max, point);
                }
            }
        }
    }
    return bounds;
}

TEST(ClusterGridTest, ShaderBoundsMatchReference)
{
    const TestCamera camera = MakeCamera();
    const ClusterGrid grid = ComputeClusterGrid(1280, 720);

    const std::uint32_t slices[] = {0, 1, 5, 12, 18, 23};
    for (const std::uint32_t slice : slices)
    {
        for (std::uint32_t tileY = 0; tileY < grid.y; tileY += 7)
        {
            for (std::uint32_t tileX = 0; tileX < grid.x; tileX += 9)
            {
                const Bounds shader = ShaderClusterBounds(camera, 1280, 720, tileX, tileY, slice);
                const Bounds reference = ReferenceClusterBounds(camera, 1280, 720, tileX, tileY, slice);

                const float maxValue = std::max(
                    {1.0f,
                     std::abs(reference.min.x),
                     std::abs(reference.min.y),
                     std::abs(reference.min.z),
                     std::abs(reference.max.x),
                     std::abs(reference.max.y),
                     std::abs(reference.max.z)});
                const float tolerance = 1e-3f * maxValue;

                for (int axis = 0; axis < 3; ++axis)
                {
                    EXPECT_NEAR(shader.min[axis], reference.min[axis], tolerance)
                        << "min axis " << axis << " tile " << tileX << "," << tileY << " slice " << slice;
                    EXPECT_NEAR(shader.max[axis], reference.max[axis], tolerance)
                        << "max axis " << axis << " tile " << tileX << "," << tileY << " slice " << slice;
                }
            }
        }
    }
}

TEST(ClusterGridTest, PointIsInsideItsOwnClusterBounds)
{
    const TestCamera camera = MakeCamera();
    const ClusterGrid grid = ComputeClusterGrid(1280, 720);

    const std::uint32_t width = 1280;
    const std::uint32_t height = 720;
    const float sliceCount = static_cast<float>(kClusterZSliceCount);
    const float sliceRatio = camera.farPlane / camera.nearPlane;

    const glm::mat4 inverseView = glm::inverse(camera.view);

    std::uint32_t tested = 0;
    for (std::uint32_t pixelY = 1; pixelY < height; pixelY += 37)
    {
        for (std::uint32_t pixelX = 1; pixelX < width; pixelX += 41)
        {
            const float ndcX = 2.0f * (static_cast<float>(pixelX) + 0.5f) / static_cast<float>(width) - 1.0f;
            const float ndcY = 2.0f * (static_cast<float>(pixelY) + 0.5f) / static_cast<float>(height) - 1.0f;

            // Camera ray for this pixel, kept at a known view depth.
            const glm::mat4 inverseProjection = glm::inverse(camera.projection);
            const glm::vec4 ray = inverseProjection * glm::vec4{ndcX, ndcY, 0.0f, 1.0f};
            const glm::vec3 viewDirection = glm::normalize(glm::vec3{ray} / ray.w);

            for (float depth = camera.nearPlane * 2.0f; depth < camera.farPlane * 0.4f; depth *= 1.9f)
            {
                const glm::vec3 viewPoint = viewDirection * (-depth / viewDirection.z);
                const glm::vec3 worldPoint = glm::vec3{inverseView * glm::vec4{viewPoint, 1.0f}};

                const float slice =
                    std::log(depth / camera.nearPlane) / std::log(sliceRatio) * sliceCount;
                const std::uint32_t clusterZ = std::min(
                    static_cast<std::uint32_t>(std::max(slice, 0.0f)),
                    kClusterZSliceCount - 1);
                const std::uint32_t tileX = pixelX / kClusterTileSize;
                const std::uint32_t tileY = pixelY / kClusterTileSize;

                const Bounds bounds = ShaderClusterBounds(camera, width, height, tileX, tileY, clusterZ);
                EXPECT_TRUE(
                    worldPoint.x >= bounds.min.x - 1e-3f && worldPoint.x <= bounds.max.x + 1e-3f &&
                    worldPoint.y >= bounds.min.y - 1e-3f && worldPoint.y <= bounds.max.y + 1e-3f &&
                    worldPoint.z >= bounds.min.z - 1e-3f && worldPoint.z <= bounds.max.z + 1e-3f)
                    << "pixel " << pixelX << "," << pixelY << " depth " << depth;
                ++tested;
            }
        }
    }

    EXPECT_GT(tested, 0u);
}

} // namespace
} // namespace mukki
