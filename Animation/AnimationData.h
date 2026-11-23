#pragma once

#include <memory>
#include <vector>

#include "Assets/GltfLoader.h"

namespace mukki
{

// CPU-side animation data retained with an uploaded model. Instances share this immutable data
// and keep their own pose state.
struct AnimationData
{
    std::vector<GltfNode> nodes{};
    std::vector<GltfSkin> skins{};
    std::vector<GltfAnimation> animations{};
    std::vector<int> rootNodes{};
    // Per-node morph target count (from the node's mesh); zero for nodes without morphs.
    std::vector<int> nodeMorphTargetCounts{};
    // Per-node default morph weights (mesh.weights); empty for nodes without defaults.
    std::vector<std::vector<float>> nodeDefaultWeights{};

    [[nodiscard]] bool IsAnimated() const
    {
        return !animations.empty() || !skins.empty();
    }
};

// Moves the animation-relevant parts of a loaded model into shared immutable data. Returns null
// when the model has neither skins, animations nor morph targets.
[[nodiscard]] inline std::shared_ptr<AnimationData> ExtractAnimationData(GltfModel& model)
{
    bool hasMorphs = false;
    for (const GltfPrimitive& primitive : model.primitives)
    {
        if (!primitive.morphTargets.empty())
        {
            hasMorphs = true;
            break;
        }
    }
    if (model.skins.empty() && model.animations.empty() && !hasMorphs)
    {
        return nullptr;
    }

    auto data = std::make_shared<AnimationData>();
    data->nodes = std::move(model.nodes);
    data->skins = std::move(model.skins);
    data->animations = std::move(model.animations);
    data->rootNodes = std::move(model.rootNodes);

    // Morph metadata is derived from the meshes the nodes reference (the primitives themselves
    // stay in the model for upload).
    data->nodeMorphTargetCounts.assign(data->nodes.size(), 0);
    data->nodeDefaultWeights.resize(data->nodes.size());
    for (std::size_t node = 0; node < data->nodes.size(); ++node)
    {
        const int meshIndex = data->nodes[node].meshIndex;
        if (meshIndex < 0 || meshIndex >= static_cast<int>(model.meshPrimitives.size()) ||
            model.meshPrimitives[static_cast<std::size_t>(meshIndex)].empty())
        {
            continue;
        }
        const std::uint32_t primitiveIndex =
            model.meshPrimitives[static_cast<std::size_t>(meshIndex)].front();
        const GltfPrimitive& primitive = model.primitives[primitiveIndex];
        data->nodeMorphTargetCounts[node] = static_cast<int>(primitive.morphTargets.size());
        data->nodeDefaultWeights[node] = primitive.defaultWeights;
    }
    return data;
}

} // namespace mukki
