#pragma once

#include <cstdint>
#include <vector>

namespace mukki
{

// Radius-parameterized subsurface diffusion LUT used by the screen space SSS blur. The texture is
// indexed by normalized tap distance (u in [0, 1] = d / radius) and a logarithmic scattering
// radius bucket (v, from kSubsurfaceLutRadiusMinPx to kSubsurfaceLutRadiusMaxPx). Each texel stores
// the per-channel diffusion weight; the blur divides by the accumulated weights so energy is
// preserved. This is the LUT flavor for KHR_materials_diffuse_transmission, which has no fixed
// profile index; the radius comes from the renderer's UI scale.
inline constexpr std::uint32_t kSubsurfaceLutWidth = 128;
inline constexpr std::uint32_t kSubsurfaceLutHeight = 16;
inline constexpr float kSubsurfaceLutRadiusMinPx = 2.0f;
inline constexpr float kSubsurfaceLutRadiusMaxPx = 64.0f;

// Fills `pixels` with kSubsurfaceLutWidth * kSubsurfaceLutHeight RGBA32F texels (row major).
void BuildSubsurfaceLut(std::vector<float>& pixels);

} // namespace mukki
