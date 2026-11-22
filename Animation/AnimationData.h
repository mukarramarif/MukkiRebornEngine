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

    [[nodiscard]] bool IsAnimated() const
    {
        return !animations.empty() || !skins.empty();
    }
};

// Moves the animation-relevant parts of a loaded model into shared immutable data. Returns null
// when the model has neither skins nor animations.
[[nodiscard]] inline std::shared_ptr<AnimationData> ExtractAnimationData(GltfModel& model)
{
    if (model.skins.empty() && model.animations.empty())
    {
        return nullptr;
    }

    auto data = std::make_shared<AnimationData>();
    data->nodes = std::move(model.nodes);
    data->skins = std::move(model.skins);
    data->animations = std::move(model.animations);
    data->rootNodes = std::move(model.rootNodes);
    return data;
}

} // namespace mukki
