#include "Renderer/Debug/DebugLines.h"

#include <cmath>

#include <glm/geometric.hpp>

namespace mukki
{
namespace
{

constexpr float kTwoPi = 6.28318530717958647692f;

glm::vec4 Dim(const glm::vec4& color, float factor)
{
    return glm::vec4{color.r * factor, color.g * factor, color.b * factor, color.a};
}

void OrthonormalBasis(const glm::vec3& direction, glm::vec3& axisA, glm::vec3& axisB)
{
    const glm::vec3 up =
        std::abs(direction.y) < 0.999f ? glm::vec3{0.0f, 1.0f, 0.0f} : glm::vec3{1.0f, 0.0f, 0.0f};
    axisA = glm::normalize(glm::cross(up, direction));
    axisB = glm::cross(direction, axisA);
}

} // namespace

void DebugLines::AddVertex(const glm::vec3& position, const glm::vec4& color)
{
    m_vertices.push_back(position.x);
    m_vertices.push_back(position.y);
    m_vertices.push_back(position.z);
    m_vertices.push_back(color.r);
    m_vertices.push_back(color.g);
    m_vertices.push_back(color.b);
    m_vertices.push_back(color.a);
}

void DebugLines::AddLine(const glm::vec3& from, const glm::vec3& to, const glm::vec4& color)
{
    AddVertex(from, color);
    AddVertex(to, color);
}

void DebugLines::AddCross(const glm::vec3& center, float size, const glm::vec4& color)
{
    AddLine(center - glm::vec3{size, 0.0f, 0.0f}, center + glm::vec3{size, 0.0f, 0.0f}, color);
    AddLine(center - glm::vec3{0.0f, size, 0.0f}, center + glm::vec3{0.0f, size, 0.0f}, color);
    AddLine(center - glm::vec3{0.0f, 0.0f, size}, center + glm::vec3{0.0f, 0.0f, size}, color);
}

void DebugLines::AddCircle(
    const glm::vec3& center,
    const glm::vec3& axisA,
    const glm::vec3& axisB,
    float radius,
    const glm::vec4& color,
    std::uint32_t segments)
{
    if (segments < 3)
    {
        segments = 3;
    }

    glm::vec3 previous = center + axisA * radius;
    for (std::uint32_t index = 1; index <= segments; ++index)
    {
        const float angle = kTwoPi * static_cast<float>(index) / static_cast<float>(segments);
        const glm::vec3 point =
            center + (axisA * std::cos(angle) + axisB * std::sin(angle)) * radius;
        AddLine(previous, point, color);
        previous = point;
    }
}

void DebugLines::AddSphere(
    const glm::vec3& center,
    float radius,
    const glm::vec4& color,
    std::uint32_t segments)
{
    AddCircle(center, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, radius, color, segments);
    AddCircle(center, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, radius, color, segments);
    AddCircle(center, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, radius, color, segments);
}

void DebugLines::AddCone(
    const glm::vec3& apex,
    const glm::vec3& direction,
    float range,
    float halfAngleRadians,
    const glm::vec4& color,
    std::uint32_t segments)
{
    glm::vec3 axisA{1.0f, 0.0f, 0.0f};
    glm::vec3 axisB{0.0f, 0.0f, 1.0f};
    OrthonormalBasis(direction, axisA, axisB);

    const glm::vec3 base = apex + direction * range;
    const float radius = std::tan(halfAngleRadians) * range;

    glm::vec3 previous = base + axisA * radius;
    AddLine(apex, previous, color);
    for (std::uint32_t index = 1; index <= segments; ++index)
    {
        const float angle = kTwoPi * static_cast<float>(index) / static_cast<float>(segments);
        const glm::vec3 point = base + (axisA * std::cos(angle) + axisB * std::sin(angle)) * radius;
        AddLine(previous, point, color);
        // Draw every fourth side line to keep the vertex count reasonable.
        if (index % 4 == 0)
        {
            AddLine(apex, point, Dim(color, 0.6f));
        }
        previous = point;
    }
}

void DebugLines::AddArrow(
    const glm::vec3& origin,
    const glm::vec3& direction,
    float length,
    const glm::vec4& color)
{
    const glm::vec3 tip = origin + direction * length;
    AddLine(origin, tip, color);

    glm::vec3 axisA{1.0f, 0.0f, 0.0f};
    glm::vec3 axisB{0.0f, 0.0f, 1.0f};
    OrthonormalBasis(direction, axisA, axisB);

    const float headLength = length * 0.25f;
    const glm::vec3 headBase = tip - direction * headLength;
    const float headWidth = length * 0.12f;
    for (int side = 0; side < 4; ++side)
    {
        const float angle = kTwoPi * static_cast<float>(side) / 4.0f;
        const glm::vec3 offset =
            (axisA * std::cos(angle) + axisB * std::sin(angle)) * headWidth;
        AddLine(tip, headBase + offset, color);
    }
}

void DebugLines::AddPointLight(const glm::vec3& position, float range, const glm::vec4& color)
{
    AddCross(position, 0.15f, color);
    AddSphere(position, range, Dim(color, 0.8f));
}

void DebugLines::AddSpotLight(
    const glm::vec3& position,
    const glm::vec3& direction,
    float range,
    float innerConeAngle,
    float outerConeAngle,
    const glm::vec4& color)
{
    AddCross(position, 0.15f, color);
    AddLine(position, position + direction * range, Dim(color, 0.8f));
    AddCone(position, direction, range, outerConeAngle, color);
    AddCone(position, direction, range, innerConeAngle, Dim(color, 0.55f));
}

void DebugLines::AddDirectionalLight(const glm::vec3& direction, const glm::vec4& color)
{
    glm::vec3 axisA{1.0f, 0.0f, 0.0f};
    glm::vec3 axisB{0.0f, 0.0f, 1.0f};
    OrthonormalBasis(direction, axisA, axisB);

    const glm::vec3 anchor = -direction * 3.0f;
    AddLine(anchor, anchor + direction * 3.0f, color);
    // A small ring where the arrow starts so the direction is readable from afar.
    AddCircle(anchor, axisA, axisB, 0.4f, Dim(color, 0.7f), 24);
    AddCross(anchor + direction * 3.0f, 0.2f, color);
}

void DebugLines::AddProbe(const GiProbe& probe)
{
    AddSphere(probe.position, probe.radius, probe.color);
    AddCross(probe.position, probe.radius * 0.6f, Dim(probe.color, 0.7f));
}

} // namespace mukki
