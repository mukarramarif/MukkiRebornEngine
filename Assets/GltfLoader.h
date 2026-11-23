#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace mukki
{

struct GltfVertex
{
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
    glm::vec2 uv{0.0f};
    // Skinning attributes (JOINTS_0 / WEIGHTS_0). Weights are floats.
    std::uint32_t joints[4]{0, 0, 0, 0};
    float weights[4]{0.0f, 0.0f, 0.0f, 0.0f};
};

// One glTF morph target: per-vertex position/normal deltas (TANGENT deltas are ignored; the
// engine recomputes no tangents). Arrays may be empty when the target does not define that
// attribute; entries past the array end are zero.
struct GltfMorphTarget
{
    std::vector<glm::vec3> positionDeltas{};
    std::vector<glm::vec3> normalDeltas{};
};

struct GltfPrimitive
{
    std::vector<GltfVertex> vertices{};
    std::vector<std::uint32_t> indices{};
    int materialIndex = -1;
    // Skin driving this primitive, or -1 when rigid. Set when any node referencing the mesh has
    // a skin (primitives are shared across nodes).
    int skinIndex = -1;
    // Morph targets in mesh order (all primitives of a mesh share the same targets per spec).
    std::vector<GltfMorphTarget> morphTargets{};
    // Mesh-level default weights (mesh.weights); empty when the mesh does not author them.
    std::vector<float> defaultWeights{};
};

enum class GltfWrapMode : std::uint8_t
{
    Repeat = 0,
    ClampToEdge,
    MirroredRepeat,
};

struct GltfTexture
{
    int imageIndex = -1;
    GltfWrapMode wrapS = GltfWrapMode::Repeat;
    GltfWrapMode wrapT = GltfWrapMode::Repeat;
};

struct GltfMaterial
{
    glm::vec4 baseColorFactor{1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    int baseColorTexture = -1;
    int metallicRoughnessTexture = -1;
    bool doubleSided = false;
    bool alphaMask = false;
    float alphaCutoff = 0.5f;
    bool alphaBlend = false;
    // KHR_materials_specular.
    float specularFactor = 1.0f;
    glm::vec3 specularColorFactor{1.0f};
    int specularTexture = -1;
    int specularColorTexture = -1;
    // KHR_materials_ior.
    float ior = 1.5f;
    // KHR_materials_transmission.
    float transmissionFactor = 0.0f;
    int transmissionTexture = -1;
    // KHR_materials_volume. attenuationDistance == 0 means no absorption (glTF default).
    float thicknessFactor = 0.0f;
    int thicknessTexture = -1;
    glm::vec3 attenuationColor{1.0f};
    float attenuationDistance = 0.0f;
    // KHR_materials_dispersion.
    float dispersion = 0.0f;
    // KHR_materials_diffuse_transmission. The scattering radius is derived from the volume
    // thickness factor (see the SSS passes).
    float diffuseTransmissionFactor = 0.0f;
    int diffuseTransmissionTexture = -1;
    glm::vec3 diffuseTransmissionColorFactor{1.0f};
    int diffuseTransmissionColorTexture = -1;
    // KHR_materials_sheen.
    glm::vec3 sheenColorFactor{0.0f};
    float sheenRoughnessFactor = 0.0f;
    int sheenColorTexture = -1;
    int sheenRoughnessTexture = -1;
    // Emissive. `emissiveFactor` is already scaled by KHR_materials_emissive_strength.
    glm::vec3 emissiveFactor{0.0f};
    int emissiveTexture = -1;
};

struct GltfImage
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels{};
};

struct GltfNode
{
    std::string name{};
    int parent = -1;
    std::vector<int> children{};
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
    glm::mat4 localTransform{1.0f}; // TRS (or the authored matrix when present)
    glm::mat4 worldTransform{1.0f}; // rest pose, computed at load (parent first)
    int meshIndex = -1;
    int skinIndex = -1;
};

enum class GltfAnimationPath : std::uint8_t
{
    Translation = 0,
    Rotation,
    Scale,
    Weights,
};

enum class GltfInterpolation : std::uint8_t
{
    Linear = 0,
    Step,
    CubicSpline,
};

struct GltfSkin
{
    std::string name{};
    int skeletonRoot = -1;
    std::vector<int> joints{};
    std::vector<glm::mat4> inverseBindMatrices{};
};

struct GltfAnimationSampler
{
    GltfInterpolation interpolation = GltfInterpolation::Linear;
    std::vector<float> times{};
    // LINEAR/STEP: one vec4 per key (xyz for translation/scale, xyzw quaternion for rotation).
    // CUBICSPLINE: three vec4s per key (in tangent, value, out tangent).
    std::vector<glm::vec4> values{};
    // WEIGHTS channels: packed scalar values (one per key per morph target;
    // CUBICSPLINE: three per key per target, in/value/out tangent order).
    std::vector<float> weights{};
};

struct GltfAnimationChannel
{
    int node = -1;
    GltfAnimationPath path = GltfAnimationPath::Translation;
    int sampler = -1;
    // Number of morph targets driven by a WEIGHTS channel (validated against the node's mesh).
    int weightCount = 0;
};

struct GltfAnimation
{
    std::string name{};
    float duration = 0.0f;
    std::vector<GltfAnimationChannel> channels{};
    std::vector<GltfAnimationSampler> samplers{};
};

struct GltfModel
{
    std::vector<GltfPrimitive> primitives{};
    std::vector<std::vector<std::uint32_t>> meshPrimitives{};
    std::vector<GltfMaterial> materials{};
    std::vector<GltfImage> images{};
    std::vector<GltfTexture> textures{};
    std::vector<GltfNode> nodes{};
    std::vector<GltfSkin> skins{};
    std::vector<GltfAnimation> animations{};
    std::vector<int> rootNodes{};
    glm::vec3 minBounds{0.0f};
    glm::vec3 maxBounds{0.0f};
    std::uint64_t vertexCount = 0;
    std::uint64_t triangleCount = 0;
};

[[nodiscard]] bool LoadGltfModel(const std::filesystem::path& path, GltfModel& model);

// Composes a node local transform from TRS (translation * rotation * scale).
[[nodiscard]] glm::mat4 ComposeNodeTransform(
    const glm::vec3& translation,
    const glm::quat& rotation,
    const glm::vec3& scale);

} // namespace mukki
