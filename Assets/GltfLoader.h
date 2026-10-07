#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

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
};

struct GltfPrimitive
{
    std::vector<GltfVertex> vertices{};
    std::vector<std::uint32_t> indices{};
    int materialIndex = -1;
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
    glm::mat4 worldTransform{1.0f};
    int meshIndex = -1;
};

struct GltfModel
{
    std::vector<GltfPrimitive> primitives{};
    std::vector<std::vector<std::uint32_t>> meshPrimitives{};
    std::vector<GltfMaterial> materials{};
    std::vector<GltfImage> images{};
    std::vector<GltfTexture> textures{};
    std::vector<GltfNode> nodes{};
    glm::vec3 minBounds{0.0f};
    glm::vec3 maxBounds{0.0f};
    std::uint64_t vertexCount = 0;
    std::uint64_t triangleCount = 0;
};

[[nodiscard]] bool LoadGltfModel(const std::filesystem::path& path, GltfModel& model);

} // namespace mukki
