#include "Renderer/Lighting/ShadowCascade.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/matrix.hpp>
#include <glm/vec4.hpp>

#include <gtest/gtest.h>

namespace
{

mukki::FrameView MakeViewAt(const glm::vec3& position, const glm::vec3& target)
{
    mukki::FrameView view{};
    view.position = position;
    view.view = glm::lookAt(position, target, glm::vec3{0.0f, 1.0f, 0.0f});
    view.projection = glm::perspective(view.fovYRadians, view.aspectRatio, view.nearPlane, view.farPlane);
    view.projection[1][1] *= -1.0f;
    view.viewProjection = view.projection * view.view;
    return view;
}

mukki::FrameView MakeView()
{
    return MakeViewAt(glm::vec3{4.0f, 2.5f, 5.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
}

glm::vec2 ProjectToUv(const glm::mat4& lightViewProjection, const glm::vec3& point)
{
    const glm::vec4 clip = lightViewProjection * glm::vec4{point, 1.0f};
    return glm::vec2{clip.x / clip.w * 0.5f + 0.5f, 0.5f - clip.y / clip.w * 0.5f};
}

glm::vec3 WorldPointAt(const mukki::FrameView& view, float ndcX, float ndcY, float depth)
{
    const glm::mat4 inverseViewProjection = glm::inverse(view.viewProjection);
    const glm::vec4 clip = view.projection * glm::vec4{0.0f, 0.0f, -depth, 1.0f};
    const float ndcZ = clip.z / clip.w;
    const glm::vec4 world = inverseViewProjection * glm::vec4{ndcX, ndcY, ndcZ, 1.0f};
    return glm::vec3{world} / world.w;
}

TEST(ShadowCascades, SplitsCoverRangeMonotonically)
{
    const mukki::FrameView view = MakeView();
    mukki::ShadowSettings settings{};
    settings.distance = 120.0f;

    const std::vector<mukki::ShadowCascade> cascades =
        mukki::ComputeShadowCascades(view, glm::vec3{0.0f, -1.0f, 0.0f}, settings);

    ASSERT_EQ(cascades.size(), mukki::kShadowCascadeCount);
    EXPECT_FLOAT_EQ(cascades.front().splitNear, view.nearPlane);
    EXPECT_FLOAT_EQ(cascades.back().splitFar, settings.distance);

    for (std::size_t index = 0; index < cascades.size(); ++index)
    {
        EXPECT_GT(cascades[index].splitFar, cascades[index].splitNear);
        if (index > 0)
        {
            EXPECT_FLOAT_EQ(cascades[index].splitNear, cascades[index - 1].splitFar);
        }
    }
}

TEST(ShadowCascades, UniformLambdaProducesUniformSplits)
{
    const mukki::FrameView view = MakeView();
    mukki::ShadowSettings settings{};
    settings.distance = 100.0f;
    settings.lambda = 0.0f;

    const std::vector<mukki::ShadowCascade> cascades =
        mukki::ComputeShadowCascades(view, glm::vec3{0.0f, -1.0f, 0.0f}, settings);

    ASSERT_EQ(cascades.size(), 4u);
    for (std::size_t index = 0; index < cascades.size(); ++index)
    {
        const float expected = view.nearPlane +
                               (settings.distance - view.nearPlane) *
                                   (static_cast<float>(index + 1) / 4.0f);
        EXPECT_NEAR(cascades[index].splitFar, expected, 1e-3f);
    }
}

TEST(ShadowCascades, FrustumPointsProjectInsideTheirCascade)
{
    const mukki::FrameView view = MakeView();
    mukki::ShadowSettings settings{};
    settings.distance = 80.0f;

    const std::vector<mukki::ShadowCascade> cascades =
        mukki::ComputeShadowCascades(view, glm::vec3{-0.4f, -1.0f, -0.3f}, settings);

    ASSERT_FALSE(cascades.empty());
    for (const mukki::ShadowCascade& cascade : cascades)
    {
        for (float t = 0.0f; t <= 1.0f; t += 0.25f)
        {
            const float depth = cascade.splitNear + (cascade.splitFar - cascade.splitNear) * t;
            for (float ndcX = -1.0f; ndcX <= 1.0f; ndcX += 0.5f)
            {
                for (float ndcY = -1.0f; ndcY <= 1.0f; ndcY += 0.5f)
                {
                    const glm::vec3 world = WorldPointAt(view, ndcX, ndcY, depth);
                    const glm::vec4 lightClip =
                        cascade.lightViewProjection * glm::vec4{world, 1.0f};
                    ASSERT_GT(lightClip.w, 0.0f);
                    const glm::vec3 ndc = glm::vec3{lightClip} / lightClip.w;

                    // Texel snapping can shift the box by up to half a texel per axis.
                    const float xyTolerance = 2.0f / static_cast<float>(mukki::kShadowResolution);
                    EXPECT_GE(ndc.x, -1.0f - xyTolerance);
                    EXPECT_LE(ndc.x, 1.0f + xyTolerance);
                    EXPECT_GE(ndc.y, -1.0f - xyTolerance);
                    EXPECT_LE(ndc.y, 1.0f + xyTolerance);
                    EXPECT_GE(ndc.z, -0.001f);
                    EXPECT_LE(ndc.z, 1.001f);
                }
            }
        }
    }
}

TEST(ShadowCascades, RelativeProjectionIsInvariantUnderCameraMotion)
{
    const glm::vec3 position{4.0f, 2.5f, 5.0f};
    const glm::vec3 target{0.0f, 1.0f, 0.0f};
    const glm::vec3 translation{0.37f, 0.0f, -0.21f};

    const glm::mat4 yaw = glm::rotate(glm::mat4{1.0f}, glm::radians(25.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::vec3 rotatedTarget = position + glm::vec3{yaw * glm::vec4{target - position, 0.0f}};

    const mukki::FrameView base = MakeViewAt(position, target);
    const mukki::FrameView translated = MakeViewAt(position + translation, target + translation);
    const mukki::FrameView rotated = MakeViewAt(position, rotatedTarget);

    mukki::ShadowSettings settings{};
    const glm::vec3 lightDirection = glm::normalize(glm::vec3{-0.4f, -1.0f, -0.3f});

    const std::vector<mukki::ShadowCascade> baseCascades =
        mukki::ComputeShadowCascades(base, lightDirection, settings);
    const std::vector<mukki::ShadowCascade> translatedCascades =
        mukki::ComputeShadowCascades(translated, lightDirection, settings);
    const std::vector<mukki::ShadowCascade> rotatedCascades =
        mukki::ComputeShadowCascades(rotated, lightDirection, settings);

    const glm::vec3 firstPoint{0.0f, 0.4f, 0.0f};
    const glm::vec3 secondPoint{0.25f, 0.0f, -0.15f};

    ASSERT_EQ(baseCascades.size(), translatedCascades.size());
    ASSERT_EQ(baseCascades.size(), rotatedCascades.size());
    for (std::size_t index = 0; index < baseCascades.size(); ++index)
    {
        const glm::mat4& baseMatrix = baseCascades[index].lightViewProjection;
        const glm::mat4& translatedMatrix = translatedCascades[index].lightViewProjection;
        const glm::mat4& rotatedMatrix = rotatedCascades[index].lightViewProjection;

        const glm::vec2 baseDelta =
            ProjectToUv(baseMatrix, firstPoint) - ProjectToUv(baseMatrix, secondPoint);
        const glm::vec2 translatedDelta =
            ProjectToUv(translatedMatrix, firstPoint) - ProjectToUv(translatedMatrix, secondPoint);
        const glm::vec2 rotatedDelta =
            ProjectToUv(rotatedMatrix, firstPoint) - ProjectToUv(rotatedMatrix, secondPoint);

        EXPECT_NEAR(baseDelta.x, translatedDelta.x, 5e-4f) << "cascade " << index;
        EXPECT_NEAR(baseDelta.y, translatedDelta.y, 5e-4f) << "cascade " << index;
        EXPECT_NEAR(baseDelta.x, rotatedDelta.x, 5e-4f) << "cascade " << index;
        EXPECT_NEAR(baseDelta.y, rotatedDelta.y, 5e-4f) << "cascade " << index;
    }
}

TEST(ShadowCascades, SpotProjectionCoversTheCone)
{
    mukki::Light light{};
    light.type = mukki::LightType::Spot;
    light.position = glm::vec3{0.0f, 4.0f, 0.0f};
    light.direction = glm::vec3{0.0f, -1.0f, 0.0f};
    light.range = 12.0f;
    light.outerConeAngle = 0.5f;

    const glm::mat4 matrix = mukki::ComputeSpotShadowMatrix(light, mukki::kShadowNearPlane);

    const auto project = [&matrix](const glm::vec3& point) {
        const glm::vec4 clip = matrix * glm::vec4{point, 1.0f};
        EXPECT_GT(clip.w, 0.0f);
        return glm::vec3{clip} / clip.w;
    };

    const glm::vec3 onAxis = light.position + light.direction * 6.0f;
    const glm::vec3 axisNdc = project(onAxis);
    EXPECT_NEAR(axisNdc.x, 0.0f, 1e-3f);
    EXPECT_NEAR(axisNdc.y, 0.0f, 1e-3f);
    EXPECT_GT(axisNdc.z, 0.0f);
    EXPECT_LT(axisNdc.z, 1.0f);

    const float coneAngle = light.outerConeAngle;
    const glm::vec3 edgeDirection{glm::sin(coneAngle), -glm::cos(coneAngle), 0.0f};
    const glm::vec3 edgeNdc = project(light.position + edgeDirection * 6.0f);
    EXPECT_GE(edgeNdc.x, -1.0f);
    EXPECT_LE(edgeNdc.x, 1.0f);
    EXPECT_GE(edgeNdc.y, -1.0f);
    EXPECT_LE(edgeNdc.y, 1.0f);

    const glm::vec3 outsideDirection{glm::sin(coneAngle * 1.6f), -glm::cos(coneAngle * 1.6f), 0.0f};
    const glm::vec3 outsideNdc = project(light.position + outsideDirection * 6.0f);
    EXPECT_GT(std::abs(outsideNdc.x), 1.0f);
}

TEST(ShadowCascades, PointFacesProjectAlongTheirAxes)
{
    mukki::Light light{};
    light.type = mukki::LightType::Point;
    light.position = glm::vec3{1.0f, 3.0f, -2.0f};
    light.range = 10.0f;

    const std::array<glm::mat4, mukki::kPointShadowFaceCount> matrices =
        mukki::ComputePointShadowMatrices(light, mukki::kShadowNearPlane);

    constexpr glm::vec3 kDirections[mukki::kPointShadowFaceCount] = {
        {1.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f}};

    for (std::uint32_t face = 0; face < mukki::kPointShadowFaceCount; ++face)
    {
        const glm::vec3 point = light.position + kDirections[face] * 4.0f;
        const glm::vec4 clip = matrices[face] * glm::vec4{point, 1.0f};
        ASSERT_GT(clip.w, 0.0f) << "face " << face;
        const glm::vec3 ndc = glm::vec3{clip} / clip.w;
        EXPECT_NEAR(ndc.x, 0.0f, 1e-3f) << "face " << face;
        EXPECT_NEAR(ndc.y, 0.0f, 1e-3f) << "face " << face;
        EXPECT_GT(ndc.z, 0.0f) << "face " << face;
        EXPECT_LT(ndc.z, 1.0f) << "face " << face;
    }
}

TEST(PointShadows, UpdatePolicyTracksLightAndGeometryChanges)
{
    mukki::PointShadowSlotState state{};
    const glm::vec3 position{1.0f, 2.0f, 3.0f};
    constexpr float kRange = 12.0f;
    constexpr std::uint32_t kRevision = 7u;

    // A fresh slot always renders.
    EXPECT_TRUE(mukki::PointShadowNeedsUpdate(state, position, kRange, kRevision));
    state = mukki::PointShadowSlotState{position, kRange, kRevision, true};

    // A cached map is reused while nothing changes.
    EXPECT_FALSE(mukki::PointShadowNeedsUpdate(state, position, kRange, kRevision));

    // Moving the light or changing its range invalidates the map.
    EXPECT_TRUE(mukki::PointShadowNeedsUpdate(state, glm::vec3{1.1f, 2.0f, 3.0f}, kRange, kRevision));
    EXPECT_TRUE(mukki::PointShadowNeedsUpdate(state, position, kRange + 1.0f, kRevision));

    // Scene geometry changes invalidate it as well.
    EXPECT_TRUE(mukki::PointShadowNeedsUpdate(state, position, kRange, kRevision + 1u));
}

TEST(ShadowCascades, ClampsCascadeCountAndIsDeterministic)
{
    const mukki::FrameView view = MakeView();
    const glm::vec3 lightDirection{0.0f, -1.0f, 0.0f};

    mukki::ShadowSettings low{};
    low.cascadeCount = 0;
    EXPECT_EQ(mukki::ComputeShadowCascades(view, lightDirection, low).size(), 1u);

    mukki::ShadowSettings high{};
    high.cascadeCount = 64;
    EXPECT_EQ(
        mukki::ComputeShadowCascades(view, lightDirection, high).size(),
        mukki::kShadowCascadeCount);

    mukki::ShadowSettings settings{};
    const std::vector<mukki::ShadowCascade> first =
        mukki::ComputeShadowCascades(view, lightDirection, settings);
    const std::vector<mukki::ShadowCascade> second =
        mukki::ComputeShadowCascades(view, lightDirection, settings);
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t cascade = 0; cascade < first.size(); ++cascade)
    {
        for (int column = 0; column < 4; ++column)
        {
            for (int row = 0; row < 4; ++row)
            {
                EXPECT_FLOAT_EQ(
                    first[cascade].lightViewProjection[column][row],
                    second[cascade].lightViewProjection[column][row]);
            }
        }
    }
}

} // namespace
