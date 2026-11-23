#include "Terrain/TerrainMesh.h"

#include "Utilities/Assert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace mukki
{

GltfModel BuildTerrainModel(const TerrainHeightfield& heightfield)
{
    MRE_ASSERT(!heightfield.Empty(), "Terrain mesh requires a generated heightfield");
    MRE_ASSERT(heightfield.resolution > 0u, "Terrain mesh requires at least one cell");

    GltfModel model{};
    if (heightfield.Empty() || heightfield.resolution == 0u)
    {
        return model;
    }

    const std::uint32_t resolution = heightfield.resolution;
    const std::uint32_t side = heightfield.SideVertices();
    const std::size_t vertexCount = heightfield.VertexCount();
    const bool hasNormals = heightfield.normals.size() == vertexCount;

    GltfPrimitive primitive{};
    primitive.vertices.resize(vertexCount);
    primitive.indices.resize(static_cast<std::size_t>(resolution) * resolution * 6u);
    primitive.materialIndex = 0;

    const float halfSize = heightfield.settings.sizeMeters * 0.5f;
    const float inverseResolution = 1.0f / static_cast<float>(resolution);

    float minHeight = heightfield.heights.empty() ? 0.0f : heightfield.heights[0];
    float maxHeight = minHeight;
    for (std::uint32_t z = 0; z <= resolution; ++z)
    {
        for (std::uint32_t x = 0; x <= resolution; ++x)
        {
            const std::size_t index = heightfield.Index(x, z);
            const float height = heightfield.heights[index];
            minHeight = std::min(minHeight, height);
            maxHeight = std::max(maxHeight, height);

            GltfVertex& vertex = primitive.vertices[index];
            vertex.position = glm::vec3{
                (static_cast<float>(x) * inverseResolution) * heightfield.settings.sizeMeters -
                    halfSize,
                height,
                (static_cast<float>(z) * inverseResolution) * heightfield.settings.sizeMeters -
                    halfSize};
            vertex.normal = hasNormals ? heightfield.normals[index] : glm::vec3{0.0f, 1.0f, 0.0f};
            vertex.uv = glm::vec2{
                static_cast<float>(x) * inverseResolution,
                static_cast<float>(z) * inverseResolution};
            vertex.joints[0] = 0u;
            vertex.joints[1] = 0u;
            vertex.joints[2] = 0u;
            vertex.joints[3] = 0u;
            vertex.weights[0] = 1.0f;
            vertex.weights[1] = 0.0f;
            vertex.weights[2] = 0.0f;
            vertex.weights[3] = 0.0f;
        }
    }

    std::size_t indexOffset = 0;
    for (std::uint32_t z = 0; z < resolution; ++z)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const std::uint32_t a = static_cast<std::uint32_t>(heightfield.Index(x, z));
            const std::uint32_t b = static_cast<std::uint32_t>(heightfield.Index(x + 1u, z));
            const std::uint32_t c = static_cast<std::uint32_t>(heightfield.Index(x, z + 1u));
            const std::uint32_t d = static_cast<std::uint32_t>(heightfield.Index(x + 1u, z + 1u));

            // Counter-clockwise seen from above so the default back-face culling keeps the top.
            primitive.indices[indexOffset++] = a;
            primitive.indices[indexOffset++] = c;
            primitive.indices[indexOffset++] = b;
            primitive.indices[indexOffset++] = b;
            primitive.indices[indexOffset++] = c;
            primitive.indices[indexOffset++] = d;
        }
    }

    model.primitives.push_back(std::move(primitive));
    model.meshPrimitives.push_back({0u});

    GltfNode node{};
    node.name = "Terrain";
    node.meshIndex = 0;
    node.worldTransform = glm::mat4{1.0f};
    model.nodes.push_back(std::move(node));
    model.rootNodes.push_back(0);

    GltfMaterial material{};
    material.baseColorFactor = glm::vec4{1.0f};
    material.metallicFactor = 0.0f;
    material.roughnessFactor = 0.92f;
    material.baseColorTexture = 0;
    model.materials.push_back(material);

    GltfTexture texture{};
    texture.imageIndex = 0;
    texture.wrapS = GltfWrapMode::ClampToEdge;
    texture.wrapT = GltfWrapMode::ClampToEdge;
    model.textures.push_back(texture);

    GltfImage image{};
    image.width = side;
    image.height = side;
    image.pixels = heightfield.albedo;
    model.images.push_back(std::move(image));

    model.minBounds = glm::vec3{-halfSize, minHeight, -halfSize};
    model.maxBounds = glm::vec3{halfSize, maxHeight, halfSize};
    model.vertexCount = vertexCount;
    model.triangleCount = static_cast<std::uint64_t>(resolution) * resolution * 2u;
    return model;
}

} // namespace mukki
