#include "Renderer/Meshlets/MeshletBuilder.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <meshoptimizer.h>

#include "Utilities/Assert.h"

namespace mukki
{

MeshletMesh BuildMeshlets(std::span<const GltfVertex> vertices, std::span<const std::uint32_t> indices)
{
    MeshletMesh result{};
    MRE_ASSERT(indices.size() % 3 == 0, "Meshlets require a triangle list");

    const std::size_t triangleCount = indices.size() / 3;
    if (triangleCount == 0 || vertices.empty())
    {
        return result;
    }

    // GltfVertex stores the position in the first 12 bytes, matching meshoptimizer's expectation.
    const float* positions = reinterpret_cast<const float*>(vertices.data());
    const std::size_t positionStride = sizeof(GltfVertex);

    const std::size_t maxMeshlets =
        meshopt_buildMeshletsBound(indices.size(), kMaxMeshletVertices, kMaxMeshletTriangles);
    std::vector<meshopt_Meshlet> sourceMeshlets(maxMeshlets);
    std::vector<std::uint32_t> sourceVertices(indices.size());
    std::vector<std::uint8_t> sourceTriangles(indices.size());

    const std::size_t meshletCount = meshopt_buildMeshlets(
        sourceMeshlets.data(),
        sourceVertices.data(),
        sourceTriangles.data(),
        indices.data(),
        indices.size(),
        positions,
        vertices.size(),
        positionStride,
        kMaxMeshletVertices,
        kMaxMeshletTriangles,
        kMeshletConeWeight);
    sourceMeshlets.resize(meshletCount);
    if (meshletCount == 0)
    {
        return result;
    }

    std::size_t vertexTotal = 0;
    for (const meshopt_Meshlet& source : sourceMeshlets)
    {
        vertexTotal = std::max<std::size_t>(
            vertexTotal,
            static_cast<std::size_t>(source.vertex_offset) + source.vertex_count);
    }
    sourceVertices.resize(vertexTotal);

    result.meshlets.reserve(meshletCount);
    result.meshletVertices = std::move(sourceVertices);
    result.meshletTriangles.reserve(triangleCount);

    for (const meshopt_Meshlet& source : sourceMeshlets)
    {
        const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            &result.meshletVertices[source.vertex_offset],
            &sourceTriangles[source.triangle_offset],
            source.triangle_count,
            positions,
            vertices.size(),
            positionStride);

        Meshlet meshlet{};
        meshlet.bounds.center = glm::vec3{bounds.center[0], bounds.center[1], bounds.center[2]};
        meshlet.bounds.radius = bounds.radius;
        // Degenerate cones come back zero initialized and never cull; meshoptimizer otherwise
        // stores a normalized axis and cutoff = sin(half angle), with 1.0 disabling culling.
        meshlet.bounds.coneAxis = glm::vec3{bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]};
        meshlet.bounds.coneCutoff = bounds.cone_cutoff;
        meshlet.vertexOffset = source.vertex_offset;
        meshlet.vertexCount = source.vertex_count;
        meshlet.triangleOffset = static_cast<std::uint32_t>(result.meshletTriangles.size());
        meshlet.triangleCount = source.triangle_count;
        MRE_ASSERT(meshlet.vertexCount <= kMaxMeshletVertices, "Meshlet vertex budget exceeded");
        MRE_ASSERT(meshlet.triangleCount <= kMaxMeshletTriangles, "Meshlet triangle budget exceeded");

        // meshoptimizer stores the micro indices as a uint8 array (3 bytes per triangle); pack
        // them into one uint32 per triangle for the GPU.
        for (std::uint32_t triangle = 0; triangle < source.triangle_count; ++triangle)
        {
            const std::size_t base = static_cast<std::size_t>(source.triangle_offset) + triangle * 3;
            const std::uint32_t i0 = sourceTriangles[base];
            const std::uint32_t i1 = sourceTriangles[base + 1];
            const std::uint32_t i2 = sourceTriangles[base + 2];
            MRE_ASSERT(i0 < 256 && i1 < 256 && i2 < 256, "Meshlet micro indices must fit into 8 bits");
            result.meshletTriangles.push_back(i0 | (i1 << 8u) | (i2 << 16u));
        }
        result.meshlets.push_back(meshlet);
    }

    return result;
}

} // namespace mukki
