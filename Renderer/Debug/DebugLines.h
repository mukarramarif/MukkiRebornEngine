#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Renderer/GlobalIllumination/GiTypes.h"

namespace mukki
{

// CPU-side line list builder for world space gizmos. Vertices are interleaved position (3 floats)
// and color (4 floats).
class DebugLines
{
public:
    void Clear() { m_vertices.clear(); }

    void AddLine(const glm::vec3& from, const glm::vec3& to, const glm::vec4& color);
    void AddCross(const glm::vec3& center, float size, const glm::vec4& color);
    void AddCircle(
        const glm::vec3& center,
        const glm::vec3& axisA,
        const glm::vec3& axisB,
        float radius,
        const glm::vec4& color,
        std::uint32_t segments = 48);
    void AddSphere(
        const glm::vec3& center,
        float radius,
        const glm::vec4& color,
        std::uint32_t segments = 24);
    void AddCone(
        const glm::vec3& apex,
        const glm::vec3& direction,
        float range,
        float halfAngleRadians,
        const glm::vec4& color,
        std::uint32_t segments = 32);
    void AddArrow(
        const glm::vec3& origin,
        const glm::vec3& direction,
        float length,
        const glm::vec4& color);

    void AddPointLight(const glm::vec3& position, float range, const glm::vec4& color);
    void AddSpotLight(
        const glm::vec3& position,
        const glm::vec3& direction,
        float range,
        float innerConeAngle,
        float outerConeAngle,
        const glm::vec4& color);
    void AddDirectionalLight(const glm::vec3& direction, const glm::vec4& color);
    void AddProbe(const GiProbe& probe);

    [[nodiscard]] bool Empty() const { return m_vertices.empty(); }
    [[nodiscard]] std::span<const float> VertexData() const { return m_vertices; }
    [[nodiscard]] std::uint32_t VertexCount() const
    {
        return static_cast<std::uint32_t>(m_vertices.size() / kFloatsPerVertex);
    }

    static constexpr std::uint32_t kFloatsPerVertex = 7;

private:
    void AddVertex(const glm::vec3& position, const glm::vec4& color);

    std::vector<float> m_vertices{};
};

} // namespace mukki
