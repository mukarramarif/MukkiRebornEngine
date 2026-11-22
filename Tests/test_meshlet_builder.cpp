#include "Renderer/Meshlets/MeshletBuilder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <glm/geometric.hpp>

namespace
{

using TriangleKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;

mukki::GltfVertex MakeVertex(float x, float y, float z, float u, float v)
{
    mukki::GltfVertex vertex{};
    vertex.position = glm::vec3{x, y, z};
    vertex.normal = glm::vec3{0.0f, 1.0f, 0.0f};
    vertex.uv = glm::vec2{u, v};
    return vertex;
}

mukki::GltfModel BuildGrid(std::uint32_t cells)
{
    mukki::GltfModel model{};
    mukki::GltfPrimitive& primitive = model.primitives.emplace_back();

    const std::uint32_t side = cells + 1;
    for (std::uint32_t y = 0; y < side; ++y)
    {
        for (std::uint32_t x = 0; x < side; ++x)
        {
            primitive.vertices.push_back(MakeVertex(
                static_cast<float>(x),
                0.0f,
                static_cast<float>(y),
                static_cast<float>(x) / static_cast<float>(cells),
                static_cast<float>(y) / static_cast<float>(cells)));
        }
    }

    for (std::uint32_t y = 0; y < cells; ++y)
    {
        for (std::uint32_t x = 0; x < cells; ++x)
        {
            const std::uint32_t v00 = y * side + x;
            const std::uint32_t v10 = v00 + 1;
            const std::uint32_t v01 = v00 + side;
            const std::uint32_t v11 = v01 + 1;
            primitive.indices.insert(primitive.indices.end(), {v00, v01, v10, v10, v01, v11});
        }
    }
    return model;
}

TEST(MeshletBuilder, EmptyInputProducesNoMeshlets)
{
    const mukki::MeshletMesh meshlets = mukki::BuildMeshlets({}, {});
    EXPECT_TRUE(meshlets.Empty());
    EXPECT_TRUE(meshlets.meshletVertices.empty());
    EXPECT_TRUE(meshlets.meshletTriangles.empty());
}

TEST(MeshletBuilder, EveryTriangleBelongsToExactlyOneMeshlet)
{
    const mukki::GltfModel model = BuildGrid(16);
    const mukki::GltfPrimitive& primitive = model.primitives.front();
    ASSERT_FALSE(primitive.indices.empty());

    const mukki::MeshletMesh meshlets =
        mukki::BuildMeshlets(primitive.vertices, primitive.indices);
    ASSERT_FALSE(meshlets.Empty());

    std::map<TriangleKey, std::uint32_t> coverage{};
    for (const mukki::Meshlet& meshlet : meshlets.meshlets)
    {
        ASSERT_LE(meshlet.vertexCount, mukki::kMaxMeshletVertices);
        ASSERT_LE(meshlet.triangleCount, mukki::kMaxMeshletTriangles);
        ASSERT_LE(meshlet.vertexOffset + meshlet.vertexCount, meshlets.meshletVertices.size());
        ASSERT_LE(meshlet.triangleOffset + meshlet.triangleCount, meshlets.meshletTriangles.size());

        for (std::uint32_t triangle = 0; triangle < meshlet.triangleCount; ++triangle)
        {
            const std::uint32_t packed = meshlets.meshletTriangles[meshlet.triangleOffset + triangle];
            const std::uint32_t i0 = packed & 0xffu;
            const std::uint32_t i1 = (packed >> 8u) & 0xffu;
            const std::uint32_t i2 = (packed >> 16u) & 0xffu;
            ASSERT_LT(i0, meshlet.vertexCount);
            ASSERT_LT(i1, meshlet.vertexCount);
            ASSERT_LT(i2, meshlet.vertexCount);

            const TriangleKey key{
                meshlets.meshletVertices[meshlet.vertexOffset + i0],
                meshlets.meshletVertices[meshlet.vertexOffset + i1],
                meshlets.meshletVertices[meshlet.vertexOffset + i2]};
            ++coverage[key];
        }
    }

    const std::size_t triangleCount = primitive.indices.size() / 3;
    std::map<TriangleKey, std::uint32_t> expected{};
    for (std::size_t triangle = 0; triangle < triangleCount; ++triangle)
    {
        ++expected[TriangleKey{
            primitive.indices[triangle * 3],
            primitive.indices[triangle * 3 + 1],
            primitive.indices[triangle * 3 + 2]}];
    }

    EXPECT_EQ(coverage, expected);
}

TEST(MeshletBuilder, BoundsContainEveryMeshletVertex)
{
    const mukki::GltfModel model = BuildGrid(8);
    const mukki::GltfPrimitive& primitive = model.primitives.front();
    const mukki::MeshletMesh meshlets =
        mukki::BuildMeshlets(primitive.vertices, primitive.indices);
    ASSERT_FALSE(meshlets.Empty());

    for (const mukki::Meshlet& meshlet : meshlets.meshlets)
    {
        const mukki::MeshletBounds& bounds = meshlet.bounds;
        EXPECT_GT(bounds.radius, 0.0f);
        EXPECT_NEAR(glm::length(bounds.coneAxis), 1.0f, 1e-4f);

        for (std::uint32_t vertex = 0; vertex < meshlet.vertexCount; ++vertex)
        {
            const glm::vec3& position =
                primitive.vertices[meshlets.meshletVertices[meshlet.vertexOffset + vertex]].position;
            EXPECT_LE(glm::length(position - bounds.center), bounds.radius + 1e-4f);
        }

        // Every face normal must lie inside the cone: cutoff = sin(half angle), so the
        // minimum dot(normal, axis) is sqrt(1 - cutoff^2). cutoff == 1 marks degenerate cones.
        if (bounds.coneCutoff < 1.0f && glm::length(bounds.coneAxis) > 0.0f)
        {
            const float minDot = std::sqrt(std::max(0.0f, 1.0f - bounds.coneCutoff * bounds.coneCutoff));
            for (std::uint32_t triangle = 0; triangle < meshlet.triangleCount; ++triangle)
            {
                const std::uint32_t packed = meshlets.meshletTriangles[meshlet.triangleOffset + triangle];
                const glm::vec3& p0 = primitive.vertices[meshlets.meshletVertices[meshlet.vertexOffset + (packed & 0xffu)]].position;
                const glm::vec3& p1 = primitive.vertices[meshlets.meshletVertices[meshlet.vertexOffset + ((packed >> 8u) & 0xffu)]].position;
                const glm::vec3& p2 = primitive.vertices[meshlets.meshletVertices[meshlet.vertexOffset + ((packed >> 16u) & 0xffu)]].position;
                const glm::vec3 normal = glm::cross(p1 - p0, p2 - p0);
                const float length = glm::length(normal);
                ASSERT_GT(length, 0.0f);
                EXPECT_GE(glm::dot(normal / length, bounds.coneAxis), minDot - 1e-4f);
            }
        }
    }
}

TEST(MeshletBuilder, ConnectedQuadProducesASingleMeshlet)
{
    // Two triangles that share the diagonal edge through their indices form one connected cluster.
    const std::vector<mukki::GltfVertex> vertices{
        MakeVertex(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
        MakeVertex(1.0f, 0.0f, 0.0f, 1.0f, 0.0f),
        MakeVertex(0.0f, 0.0f, 1.0f, 0.0f, 1.0f),
        MakeVertex(1.0f, 0.0f, 1.0f, 1.0f, 1.0f)};
    const std::vector<std::uint32_t> indices{0, 2, 1, 1, 2, 3};

    const mukki::MeshletMesh meshlets = mukki::BuildMeshlets(vertices, indices);
    ASSERT_EQ(meshlets.meshlets.size(), 1u);
    EXPECT_EQ(meshlets.meshlets.front().triangleCount, 2u);
    EXPECT_EQ(meshlets.meshlets.front().vertexCount, 4u);
}

} // namespace
