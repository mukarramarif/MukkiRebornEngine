// Parametric environment used by the image based lighting. Must stay in sync with
// Renderer/Lighting/Environment.cpp (evaluation, SH band order and band factors).
static const float kSkyPi = 3.14159265359;

// Sky radiance of the gradient, excluding the sun disc (the analytic sun light covers that).
float3 SkyRadiance(
    float3 direction,
    float3 zenith,
    float3 horizon,
    float3 ground,
    float intensity)
{
    const float up = saturate(direction.y);
    const float down = saturate(-direction.y);
    const float3 upper = lerp(horizon, zenith, sqrt(up));
    const float3 lower = lerp(horizon, ground, sqrt(down));
    return (direction.y >= 0.0 ? upper : lower) * intensity;
}

// Irradiance / pi from the SH L2 coefficients (Lambert band factors are already folded in).
float3 EvaluateSkySh(float4 sh[9], float3 direction)
{
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    return sh[0].rgb * 0.282095 +
           sh[1].rgb * (0.488603 * x) +
           sh[2].rgb * (0.488603 * y) +
           sh[3].rgb * (0.488603 * z) +
           sh[4].rgb * (1.092548 * x * y) +
           sh[5].rgb * (1.092548 * y * z) +
           sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) +
           sh[7].rgb * (1.092548 * x * z) +
           sh[8].rgb * (0.546274 * (x * x - y * y));
}

static const float2 kSkyDiskTaps[8] = {
    float2(1.0, 0.0),
    float2(0.7071, 0.7071),
    float2(0.0, 1.0),
    float2(-0.7071, 0.7071),
    float2(-1.0, 0.0),
    float2(-0.7071, -0.7071),
    float2(0.0, -1.0),
    float2(0.7071, -0.7071)};

// Cheap prefilter approximation: a small fixed cone around the reflection direction, blended into
// the SH irradiance as roughness grows. The gradient sky is low frequency, so this is close to a
// prefiltered cubemap without the extra GPU passes.
float3 PrefilteredSky(
    float3 reflection,
    float roughness,
    float3 zenith,
    float3 horizon,
    float3 ground,
    float intensity,
    float4 sh[9])
{
    if (roughness <= 0.02)
    {
        return SkyRadiance(reflection, zenith, horizon, ground, intensity);
    }

    const float3 up = abs(reflection.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(up, reflection));
    const float3 bitangent = cross(reflection, tangent);
    const float cone = roughness * roughness * 1.5;

    float3 sum = 0.0;
    [unroll]
    for (int tap = 0; tap < 8; ++tap)
    {
        const float3 direction =
            normalize(reflection + (tangent * kSkyDiskTaps[tap].x + bitangent * kSkyDiskTaps[tap].y) * cone);
        sum += SkyRadiance(direction, zenith, horizon, ground, intensity);
    }

    const float3 blurred = sum * 0.125;
    const float3 ambient = EvaluateSkySh(sh, reflection);
    return lerp(blurred, ambient, smoothstep(0.3, 0.9, roughness));
}

// Equirectangular mapping used by loaded environment maps.
//   phi = atan2(z, x) = (u - 0.5) * 2*pi, theta = acos(y) = v * pi.
// Matches Environment.cpp.
float2 DirectionToEquirectUv(float3 direction)
{
    return float2(
        atan2(direction.z, direction.x) / (2.0 * kSkyPi) + 0.5,
        acos(clamp(direction.y, -1.0, 1.0)) / kSkyPi);
}

// Samples the bindless environment map. The includer must declare gTextures/gSamplers.
float3 SampleEnvironmentMap(float3 direction, uint mapIndex, uint samplerIndex, float intensity)
{
    return gTextures[NonUniformResourceIndex(mapIndex)]
               .SampleLevel(gSamplers[NonUniformResourceIndex(samplerIndex)], DirectionToEquirectUv(direction), 0.0)
               .rgb *
           intensity;
}

// Environment radiance for a direction: a loaded equirect map when available, the gradient sky
// otherwise.
float3 EnvironmentRadiance(
    float3 direction,
    float3 zenith,
    float3 horizon,
    float3 ground,
    float intensity,
    uint mapIndex,
    uint samplerIndex,
    bool useMap)
{
    return useMap
               ? SampleEnvironmentMap(direction, mapIndex, samplerIndex, intensity)
               : SkyRadiance(direction, zenith, horizon, ground, intensity);
}

// Rough reflections: cone sampling of the environment using the same fixed taps as the procedural
// path, blended into the SH irradiance as roughness grows.
float3 EnvironmentPrefiltered(
    float3 reflection,
    float roughness,
    float3 zenith,
    float3 horizon,
    float3 ground,
    float intensity,
    float4 sh[9],
    uint mapIndex,
    uint samplerIndex,
    bool useMap)
{
    if (!useMap)
    {
        return PrefilteredSky(reflection, roughness, zenith, horizon, ground, intensity, sh);
    }

    if (roughness <= 0.02)
    {
        return SampleEnvironmentMap(reflection, mapIndex, samplerIndex, intensity);
    }

    const float3 up = abs(reflection.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(up, reflection));
    const float3 bitangent = cross(reflection, tangent);
    const float cone = roughness * roughness * 1.5;

    float3 sum = 0.0;
    [unroll]
    for (int tap = 0; tap < 8; ++tap)
    {
        const float3 direction =
            normalize(reflection + (tangent * kSkyDiskTaps[tap].x + bitangent * kSkyDiskTaps[tap].y) * cone);
        sum += SampleEnvironmentMap(direction, mapIndex, samplerIndex, intensity);
    }

    const float3 blurred = sum * 0.125;
    const float3 ambient = EvaluateSkySh(sh, reflection);
    return lerp(blurred, ambient, smoothstep(0.3, 0.9, roughness));
}

// Visible sky background: the gradient plus a small sun disc for the sky pixels.
float3 SkyBackground(
    float3 direction,
    float3 zenith,
    float3 horizon,
    float3 ground,
    float intensity,
    float3 sunDirection,
    float3 sunColor,
    float sunDiscIntensity)
{
    const float cosAngle = dot(direction, sunDirection);
    const float disc = smoothstep(0.99992, 0.99996, cosAngle);
    return SkyRadiance(direction, zenith, horizon, ground, intensity) +
           sunColor * (sunDiscIntensity * disc);
}

// Visible sky background: the environment map or the gradient plus a small sun disc.
float3 EnvironmentBackground(
    float3 direction,
    float3 zenith,
    float3 horizon,
    float3 ground,
    float intensity,
    float3 sunDirection,
    float3 sunColor,
    float sunDiscIntensity,
    uint mapIndex,
    uint samplerIndex,
    bool useMap)
{
    if (useMap)
    {
        return SampleEnvironmentMap(direction, mapIndex, samplerIndex, intensity);
    }
    return SkyBackground(
        direction, zenith, horizon, ground, intensity, sunDirection, sunColor, sunDiscIntensity);
}
