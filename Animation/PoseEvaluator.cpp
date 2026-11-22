#include "Animation/PoseEvaluator.h"

#include <algorithm>
#include <cmath>

namespace mukki
{
namespace
{

bool FindKey(const std::vector<float>& times, float time, std::size_t& outIndex, float& outT)
{
    if (times.empty())
    {
        return false;
    }
    if (time <= times.front())
    {
        outIndex = 0;
        outT = 0.0f;
        return true;
    }
    if (time >= times.back())
    {
        outIndex = times.size() - 1;
        outT = 0.0f;
        return true;
    }

    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    const std::size_t high = static_cast<std::size_t>(upper - times.begin());
    const std::size_t low = high - 1;
    const float span = times[high] - times[low];
    outIndex = low;
    outT = span > 0.0f ? (time - times[low]) / span : 0.0f;
    return true;
}

// LINEAR/STEP store one value per key; CUBICSPLINE stores (in tangent, value, out tangent).
const glm::vec4& SampleRaw(
    const GltfAnimationSampler& sampler,
    std::size_t key,
    std::size_t stride)
{
    return sampler.values[key * stride + (stride == 3u ? 1u : 0u)];
}

glm::vec3 SampleVec3(const GltfAnimationSampler& sampler, float time)
{
    if (sampler.times.empty())
    {
        return glm::vec3{0.0f};
    }

    std::size_t low = 0;
    float t = 0.0f;
    FindKey(sampler.times, time, low, t);
    const std::size_t stride =
        sampler.interpolation == GltfInterpolation::CubicSpline ? 3u : 1u;
    const glm::vec3 a{glm::vec3{SampleRaw(sampler, low, stride)}};
    if (sampler.interpolation == GltfInterpolation::Step ||
        low + 1 >= sampler.times.size() || t <= 0.0f)
    {
        return a;
    }

    const glm::vec3 b{glm::vec3{SampleRaw(sampler, low + 1, stride)}};
    if (sampler.interpolation == GltfInterpolation::Linear)
    {
        return glm::mix(a, b, t);
    }

    // Hermite between the keys using the neighboring tangents.
    const float deltaTime = sampler.times[low + 1] - sampler.times[low];
    const glm::vec3 outTangent{sampler.values[low * stride]};
    const glm::vec3 inTangent{sampler.values[(low + 1) * stride + 2u]};
    const float t2 = t * t;
    const float t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * a +
           (t3 - 2.0f * t2 + t) * deltaTime * outTangent +
           (-2.0f * t3 + 3.0f * t2) * b +
           (t3 - t2) * deltaTime * inTangent;
}

glm::quat SampleRotation(const GltfAnimationSampler& sampler, float time)
{
    if (sampler.times.empty())
    {
        return glm::quat{1.0f, 0.0f, 0.0f, 0.0f};
    }

    std::size_t low = 0;
    float t = 0.0f;
    FindKey(sampler.times, time, low, t);
    const std::size_t stride =
        sampler.interpolation == GltfInterpolation::CubicSpline ? 3u : 1u;
    const glm::vec4 rawA = SampleRaw(sampler, low, stride);
    const glm::quat a{rawA.w, rawA.x, rawA.y, rawA.z};
    if (sampler.interpolation == GltfInterpolation::Step ||
        low + 1 >= sampler.times.size() || t <= 0.0f)
    {
        return a;
    }

    const glm::vec4 rawB = SampleRaw(sampler, low + 1, stride);
    const glm::quat b{rawB.w, rawB.x, rawB.y, rawB.z};
    // Cubic spline rotations are approximated with slerp; the loader warns about it.
    return glm::slerp(a, b, t);
}

void ComputeWorldTransforms(
    const AnimationData& data,
    const std::vector<glm::mat4>& local,
    std::vector<glm::mat4>& world)
{
    world.resize(data.nodes.size());
    const auto compute = [&data, &local, &world](
                             auto&& self,
                             int nodeIndex,
                             const glm::mat4& parentTransform) -> void {
        world[static_cast<std::size_t>(nodeIndex)] =
            parentTransform * local[static_cast<std::size_t>(nodeIndex)];
        const GltfNode& node = data.nodes[static_cast<std::size_t>(nodeIndex)];
        for (const int child : node.children)
        {
            self(self, child, world[static_cast<std::size_t>(nodeIndex)]);
        }
    };
    for (const int root : data.rootNodes)
    {
        if (root >= 0 && root < static_cast<int>(data.nodes.size()))
        {
            compute(compute, root, glm::mat4{1.0f});
        }
    }
}

void FillJointMatrices(const AnimationData& data, PoseData& pose)
{
    std::size_t totalJoints = 0;
    for (const GltfSkin& skin : data.skins)
    {
        totalJoints += skin.joints.size();
    }
    pose.jointMatrices.resize(totalJoints);

    std::size_t offset = 0;
    for (const GltfSkin& skin : data.skins)
    {
        for (std::size_t joint = 0; joint < skin.joints.size(); ++joint)
        {
            const int jointNode = skin.joints[joint];
            const glm::mat4 jointWorld =
                jointNode >= 0 && jointNode < static_cast<int>(data.nodes.size())
                    ? pose.nodeWorld[static_cast<std::size_t>(jointNode)]
                    : glm::mat4{1.0f};
            const glm::mat4 inverseBind =
                joint < skin.inverseBindMatrices.size() ? skin.inverseBindMatrices[joint]
                                                        : glm::mat4{1.0f};
            pose.jointMatrices[offset + joint] = jointWorld * inverseBind;
        }
        offset += skin.joints.size();
    }
}

} // namespace

float WrapClipTime(float duration, float time, bool loop)
{
    if (duration <= 0.0f)
    {
        return 0.0f;
    }
    if (loop)
    {
        const float wrapped = std::fmod(time, duration);
        return wrapped < 0.0f ? wrapped + duration : wrapped;
    }
    return std::clamp(time, 0.0f, duration);
}

PoseData RestPose(const AnimationData& data)
{
    PoseData pose{};
    std::vector<glm::mat4> local(data.nodes.size());
    for (std::size_t node = 0; node < data.nodes.size(); ++node)
    {
        local[node] = data.nodes[node].localTransform;
    }
    ComputeWorldTransforms(data, local, pose.nodeWorld);
    FillJointMatrices(data, pose);
    return pose;
}

bool EvaluatePose(
    const AnimationData& data,
    const GltfAnimation& clip,
    float time,
    PoseData& pose)
{
    const std::size_t nodeCount = data.nodes.size();
    std::vector<glm::vec3> translations(nodeCount);
    std::vector<glm::quat> rotations(nodeCount);
    std::vector<glm::vec3> scales(nodeCount);
    std::vector<bool> animated(nodeCount, false);
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        translations[node] = data.nodes[node].translation;
        rotations[node] = data.nodes[node].rotation;
        scales[node] = data.nodes[node].scale;
    }

    for (const GltfAnimationChannel& channel : clip.channels)
    {
        if (channel.node < 0 || channel.node >= static_cast<int>(nodeCount) ||
            channel.sampler < 0 || channel.sampler >= static_cast<int>(clip.samplers.size()))
        {
            continue;
        }
        const GltfAnimationSampler& sampler =
            clip.samplers[static_cast<std::size_t>(channel.sampler)];
        if (sampler.times.empty())
        {
            continue;
        }

        const std::size_t node = static_cast<std::size_t>(channel.node);
        switch (channel.path)
        {
        case GltfAnimationPath::Translation:
            translations[node] = SampleVec3(sampler, time);
            animated[node] = true;
            break;
        case GltfAnimationPath::Rotation:
            rotations[node] = SampleRotation(sampler, time);
            animated[node] = true;
            break;
        case GltfAnimationPath::Scale:
            scales[node] = SampleVec3(sampler, time);
            animated[node] = true;
            break;
        case GltfAnimationPath::Weights:
            // Morph targets are not supported yet.
            break;
        }
    }

    std::vector<glm::mat4> local(nodeCount);
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        // Matrix-authored nodes keep their matrix unless a channel overrides their TRS.
        local[node] = animated[node] ? ComposeNodeTransform(translations[node], rotations[node], scales[node])
                                     : data.nodes[node].localTransform;
    }

    ComputeWorldTransforms(data, local, pose.nodeWorld);
    FillJointMatrices(data, pose);
    return true;
}

} // namespace mukki
