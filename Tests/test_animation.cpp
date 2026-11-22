#include "Animation/PoseEvaluator.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include <glm/gtc/quaternion.hpp>

#include <gtest/gtest.h>

namespace
{

// node 0: root at the origin; node 1: child resting at (0, 1, 0).
mukki::AnimationData MakeSimpleHierarchy()
{
    mukki::AnimationData data{};
    data.nodes.resize(2);
    data.nodes[0].children.push_back(1);
    data.nodes[0].localTransform = glm::mat4{1.0f};
    data.nodes[1].parent = 0;
    data.nodes[1].translation = glm::vec3{0.0f, 1.0f, 0.0f};
    data.nodes[1].localTransform =
        mukki::ComposeNodeTransform(
            data.nodes[1].translation,
            data.nodes[1].rotation,
            data.nodes[1].scale);
    data.rootNodes.push_back(0);
    return data;
}

mukki::GltfAnimation MakeClip(
    mukki::GltfAnimationPath path,
    mukki::GltfInterpolation interpolation,
    const std::vector<float>& times,
    const std::vector<glm::vec4>& values)
{
    mukki::GltfAnimation clip{};
    clip.duration = times.back();
    mukki::GltfAnimationSampler sampler{};
    sampler.interpolation = interpolation;
    sampler.times = times;
    sampler.values = values;
    clip.samplers.push_back(std::move(sampler));
    clip.channels.push_back(mukki::GltfAnimationChannel{1, path, 0});
    return clip;
}

glm::quat AngleAxisQuat(float radians, const glm::vec3& axis)
{
    return glm::quat{std::cos(radians * 0.5f), axis * std::sin(radians * 0.5f)};
}

TEST(Animation, WrapClipTimeLoopsAndClamps)
{
    EXPECT_FLOAT_EQ(mukki::WrapClipTime(2.0f, 1.0f, true), 1.0f);
    EXPECT_FLOAT_EQ(mukki::WrapClipTime(2.0f, 2.0f, true), 0.0f);
    EXPECT_FLOAT_EQ(mukki::WrapClipTime(2.0f, 2.5f, true), 0.5f);
    EXPECT_FLOAT_EQ(mukki::WrapClipTime(2.0f, -0.5f, true), 1.5f);
    EXPECT_FLOAT_EQ(mukki::WrapClipTime(2.0f, 5.0f, false), 2.0f);
    EXPECT_FLOAT_EQ(mukki::WrapClipTime(0.0f, 1.0f, true), 0.0f);
}

TEST(Animation, RestPoseMatchesTheHierarchy)
{
    const mukki::AnimationData data = MakeSimpleHierarchy();
    const mukki::PoseData pose = mukki::RestPose(data);

    ASSERT_EQ(pose.nodeWorld.size(), 2u);
    EXPECT_EQ(pose.nodeWorld[0], glm::mat4{1.0f});
    const glm::mat4 expectedChild = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, 1.0f, 0.0f});
    EXPECT_EQ(pose.nodeWorld[1], expectedChild);
    EXPECT_TRUE(pose.jointMatrices.empty());
}

TEST(Animation, LinearTranslationSamplesAndEvaluates)
{
    const mukki::AnimationData data = MakeSimpleHierarchy();
    const mukki::GltfAnimation clip = MakeClip(
        mukki::GltfAnimationPath::Translation,
        mukki::GltfInterpolation::Linear,
        {0.0f, 1.0f},
        {glm::vec4{0.0f}, glm::vec4{2.0f, 0.0f, 0.0f, 0.0f}});

    mukki::PoseData pose{};
    ASSERT_TRUE(mukki::EvaluatePose(data, clip, 0.5f, pose));

    // Animated translation replaces the node's rest translation, so the child moves to (1, 0, 0).
    const glm::mat4 expected = glm::translate(
        glm::mat4{1.0f},
        glm::vec3{1.0f, 0.0f, 0.0f});
    EXPECT_EQ(pose.nodeWorld[1], expected);
}

TEST(Animation, StepInterpolationUsesTheLowerKey)
{
    const mukki::AnimationData data = MakeSimpleHierarchy();
    const mukki::GltfAnimation clip = MakeClip(
        mukki::GltfAnimationPath::Translation,
        mukki::GltfInterpolation::Step,
        {0.0f, 1.0f},
        {glm::vec4{0.0f}, glm::vec4{2.0f, 0.0f, 0.0f, 0.0f}});

    mukki::PoseData pose{};
    ASSERT_TRUE(mukki::EvaluatePose(data, clip, 0.9f, pose));
    EXPECT_EQ(pose.nodeWorld[1][3].x, 0.0f);

    ASSERT_TRUE(mukki::EvaluatePose(data, clip, 1.0f, pose));
    EXPECT_EQ(pose.nodeWorld[1][3].x, 2.0f);
}

TEST(Animation, CubicSplineInterpolatesWithTangents)
{
    const mukki::AnimationData data = MakeSimpleHierarchy();
    // Single segment 0..1: value 0 with out tangent 1, value 1 with in tangent 1.
    mukki::GltfAnimationSampler sampler{};
    sampler.interpolation = mukki::GltfInterpolation::CubicSpline;
    sampler.times = {0.0f, 1.0f};
    sampler.values = {
        glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}, // in tangent of key 0 (unused)
        glm::vec4{0.0f},                    // value 0
        glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}, // out tangent of key 0
        glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}, // in tangent of key 1
        glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}, // value 1
        glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}, // out tangent of key 1 (unused)
    };
    mukki::GltfAnimation clip{};
    clip.duration = 1.0f;
    clip.samplers.push_back(sampler);
    clip.channels.push_back(mukki::GltfAnimationChannel{1, mukki::GltfAnimationPath::Translation, 0});

    mukki::PoseData pose{};
    ASSERT_TRUE(mukki::EvaluatePose(data, clip, 0.5f, pose));

    // Hermite with both tangents 1 and delta 1: value(0.5) = 0.5.
    EXPECT_NEAR(pose.nodeWorld[1][3].x, 0.5f, 1e-5f);
}

TEST(Animation, RotationSlerpTakesTheShortestPath)
{
    const mukki::AnimationData data = MakeSimpleHierarchy();
    const glm::vec3 axis{0.0f, 1.0f, 0.0f};
    const glm::quat start = AngleAxisQuat(glm::radians(170.0f), axis);
    const glm::quat end = AngleAxisQuat(glm::radians(-170.0f), axis);
    const mukki::GltfAnimation clip = MakeClip(
        mukki::GltfAnimationPath::Rotation,
        mukki::GltfInterpolation::Linear,
        {0.0f, 1.0f},
        {glm::vec4{start.x, start.y, start.z, start.w},
         glm::vec4{end.x, end.y, end.z, end.w}});

    mukki::PoseData pose{};
    ASSERT_TRUE(mukki::EvaluatePose(data, clip, 0.5f, pose));

    // Halfway along the short (20 degree) arc is 180 degrees about Y.
    const glm::quat expected = AngleAxisQuat(glm::pi<float>(), axis);
    const glm::quat sampled = glm::quat_cast(pose.nodeWorld[1]);
    EXPECT_NEAR(std::abs(glm::dot(sampled, expected)), 1.0f, 1e-5f);
}

TEST(Animation, JointMatricesCombineWorldAndInverseBind)
{
    mukki::AnimationData data = MakeSimpleHierarchy();
    mukki::GltfSkin skin{};
    skin.joints = {1};
    skin.inverseBindMatrices = {glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, -1.0f, 0.0f})};
    data.skins.push_back(std::move(skin));

    const mukki::PoseData pose = mukki::RestPose(data);
    ASSERT_EQ(pose.jointMatrices.size(), 1u);
    // jointWorld (y=1) * inverseBind (y=-1) = identity.
    EXPECT_EQ(pose.jointMatrices[0], glm::mat4{1.0f});
}

TEST(Animation, RiggedSimpleTimeZeroMatchesTheRestPose)
{
    const std::filesystem::path modelPath =
        std::filesystem::path{MRE_SOURCE_DIR} /
        "ThirdParty/glTF-Sample-Models/2.0/RiggedSimple/glTF/RiggedSimple.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));
    auto animationData = mukki::ExtractAnimationData(model);
    ASSERT_NE(animationData, nullptr);
    ASSERT_FALSE(animationData->animations.empty());

    const mukki::PoseData rest = mukki::RestPose(*animationData);
    const mukki::GltfAnimation& clip = animationData->animations.front();

    mukki::PoseData atZero{};
    ASSERT_TRUE(mukki::EvaluatePose(*animationData, clip, 0.0f, atZero));
    EXPECT_EQ(atZero.nodeWorld.size(), rest.nodeWorld.size());
    for (std::size_t node = 0; node < rest.nodeWorld.size(); ++node)
    {
        for (int column = 0; column < 4; ++column)
        {
            for (int row = 0; row < 4; ++row)
            {
                EXPECT_NEAR(
                    atZero.nodeWorld[node][column][row],
                    rest.nodeWorld[node][column][row],
                    1e-4f);
            }
        }
    }

    mukki::PoseData atHalf{};
    ASSERT_TRUE(mukki::EvaluatePose(*animationData, clip, clip.duration * 0.5f, atHalf));
    bool differs = false;
    for (std::size_t node = 0; node < rest.nodeWorld.size(); ++node)
    {
        if (atHalf.nodeWorld[node] != rest.nodeWorld[node])
        {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs);
}

TEST(Animation, FoxSkinDeformsOverTime)
{
    const std::filesystem::path modelPath =
        std::filesystem::path{MRE_SOURCE_DIR} /
        "ThirdParty/glTF-Sample-Models/2.0/Fox/glTF/Fox.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    // The Fox primitive is skinned and its vertices carry joint/weight attributes.
    ASSERT_FALSE(model.primitives.empty());
    EXPECT_EQ(model.primitives[0].skinIndex, 0);
    const mukki::GltfVertex* vertex = nullptr;
    for (const mukki::GltfVertex& candidate : model.primitives[0].vertices)
    {
        if (candidate.weights[0] > 0.0f)
        {
            vertex = &candidate;
            break;
        }
    }
    ASSERT_NE(vertex, nullptr);

    auto animationData = mukki::ExtractAnimationData(model);
    ASSERT_NE(animationData, nullptr);
    ASSERT_EQ(animationData->skins.size(), 1u);
    ASSERT_FALSE(animationData->animations.empty());

    const auto survey = std::find_if(
        animationData->animations.begin(),
        animationData->animations.end(),
        [](const mukki::GltfAnimation& clip) { return clip.name == "Survey"; });
    ASSERT_NE(survey, animationData->animations.end());

    const mukki::PoseData rest = mukki::RestPose(*animationData);
    const std::size_t jointCount = animationData->skins[0].joints.size();
    ASSERT_EQ(rest.jointMatrices.size(), jointCount);
    ASSERT_GT(jointCount, 0u);

    // Every joint matrix must be finite: a single NaN would poison the GPU palette.
    for (const glm::mat4& matrix : rest.jointMatrices)
    {
        for (int column = 0; column < 4; ++column)
        {
            for (int row = 0; row < 4; ++row)
            {
                EXPECT_TRUE(std::isfinite(matrix[column][row]));
            }
        }
    }

    // The sampled vertex references only joints the skin owns.
    for (int influence = 0; influence < 4; ++influence)
    {
        if (vertex->weights[influence] > 0.0f)
        {
            ASSERT_LT(vertex->joints[influence], jointCount);
        }
    }

    const auto skinVertex = [&](const mukki::PoseData& pose) {
        const glm::vec4 local{vertex->position, 1.0f};
        glm::vec4 skinned{0.0f};
        for (int influence = 0; influence < 4; ++influence)
        {
            const float weight = vertex->weights[influence];
            if (weight <= 0.0f)
            {
                continue;
            }
            skinned +=
                pose.jointMatrices[vertex->joints[influence]] * local * weight;
        }
        return glm::vec3{skinned};
    };

    const glm::vec3 restPosition = skinVertex(rest);

    mukki::PoseData atOne{};
    ASSERT_TRUE(mukki::EvaluatePose(*animationData, *survey, 1.0f, atOne));
    const glm::vec3 animatedPosition = skinVertex(atOne);

    // One second into the Survey clip the fox has clearly moved.
    EXPECT_GT(glm::distance(restPosition, animatedPosition), 1e-3f);
}

} // namespace
