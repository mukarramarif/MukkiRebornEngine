#pragma once

#include <vector>

#include <glm/mat4x4.hpp>

#include "Animation/AnimationData.h"

namespace mukki
{

// Evaluated pose of one model instance: world transforms in model space, the flattened joint
// matrices (jointWorld * inverseBind) ordered by skin and the per-node morph weights.
struct PoseData
{
    std::vector<glm::mat4> nodeWorld{};
    std::vector<glm::mat4> jointMatrices{};
    // Per-node morph target weights (indexed by node, then target).
    std::vector<std::vector<float>> nodeWeights{};
};

// The pose with every node at its rest transform.
[[nodiscard]] PoseData RestPose(const AnimationData& data);

// Samples `clip` at `time` (already wrapped/clamped by the caller) and evaluates the full pose.
// Returns false when the clip has no usable channels.
[[nodiscard]] bool EvaluatePose(
    const AnimationData& data,
    const GltfAnimation& clip,
    float time,
    PoseData& pose);

// Wraps (loop) or clamps a clip time against its duration.
[[nodiscard]] float WrapClipTime(float duration, float time, bool loop);

} // namespace mukki
