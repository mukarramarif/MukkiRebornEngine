#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <glm/vec3.hpp>

#include "Assets/GltfLoader.h"

namespace mukki
{

// Meshlet limits. Micro triangle indices are packed into 3x8 bits, so the vertex budget must stay
// below 256. Both budgets stay inside the guaranteed mesh shader output limits (256 vertices and
// 256 primitives per workgroup).
inline constexpr std::uint32_t kMaxMeshletVertices = 64;
inline constexpr std::uint32_t kMaxMeshletTriangles = 124;
// Cluster growth bias toward tight normal cones (0 = size only, 1 = cone only).
inline constexpr float kMeshletConeWeight = 0.5f;

struct MeshletBounds
{
    glm::vec3 center{0.0f};
    float radius = 0.0f;
    glm::vec3 coneAxis{0.0f, 1.0f, 0.0f};
    // meshoptimizer normal cone convention: the axis bounds the face normals and the cutoff is
    // sin(half angle). A meshlet is back-facing when
    //   dot(normalize(center - camera), axis) >= cutoff + radius / distance(center, camera).
    // cutoff == 1 (or a zero axis) disables backface culling.
    float coneCutoff = 1.0f;
};

struct Meshlet
{
    MeshletBounds bounds{};
    std::uint32_t vertexOffset = 0;   // base into MeshletMesh::meshletVertices
    std::uint32_t triangleOffset = 0; // base into MeshletMesh::meshletTriangles
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
};

// A triangle list split into clusters. `meshletVertices` stores the global (source) vertex
// indices; `meshletTriangles` stores one packed triangle per entry with the local indices in
// bytes 0, 1 and 2.
struct MeshletMesh
{
    std::vector<Meshlet> meshlets{};
    std::vector<std::uint32_t> meshletVertices{};
    std::vector<std::uint32_t> meshletTriangles{};

    [[nodiscard]] bool Empty() const { return meshlets.empty(); }
};

// Splits a triangle list into connected meshlets of at most kMaxMeshletVertices unique vertices and
// kMaxMeshletTriangles triangles. Every input triangle appears in exactly one meshlet.
[[nodiscard]] MeshletMesh BuildMeshlets(
    std::span<const GltfVertex> vertices,
    std::span<const std::uint32_t> indices);

} // namespace mukki
