struct SceneUniform
{
    float4x4 viewProjection;
    float4x4 previousViewProjection;
    float4x4 view;
    float4x4 inverseViewProjection;
    float4 cameraPosition;
    float4 ambientColor;
    float4 clusterParams; // x: near, y: far, z: tile size in pixels, w: z slice count
    uint4 clusterGrid;    // x, y, z grid dimensions, w: light count
    float4 screenSize;
    float4 reflectionParams; // x: reflection intensity
    float4 sunDirection;     // xyz: direction toward the sun
    float4 sunColor;         // rgb: color * intensity
    float4 skySh[9];         // environment irradiance / pi SH L2
    float4 skyParams;        // x: enabled, y: intensity, z: sun disc, w: unused
    float4 skyZenith;
    float4 skyHorizon;
    float4 skyGround;
    float4 skyMap;    // x: bindless index, y: sampler index, z: has map, w: unused
};

struct GpuLight
{
    float4 positionRange;
    float4 directionIntensity;
    float4 color;
    float4 params; // x: type, y: inner cone cosine, z: outer cone cosine
};

// Must match mukki::kMaxShadowViews (4 cascades + 4 spots + 4 point lights * 6 faces).
static const uint kMaxShadowViews = 32;
static const uint kMaxPointShadowLights = 4;

struct ShadowUniform
{
    float4x4 viewProjections[kMaxShadowViews];
    float4 cascadeSplits;
    float4 params;   // x: depthBias, y: normalBias, z: distance, w: enabled
    float4 rtParams; // x: ray traced sun shadows enabled, y/z/w: unused
};

struct GpuGiProbe
{
    float4 position; // xyz: world position
    float4 sh0;      // L0 rgb
    float4 shX;
    float4 shY;
    float4 shZ;
    float4 depth;    // x: mean hit distance, y: mean squared, z: valid, w: backface fraction
};

// Must match DeferredRenderer::GpuObjectData.
struct GpuObjectData
{
    float4x4 previousModel;
    float4 specular;
    uint4 specularTextureIndices;
    uint4 specularSamplerIndices;
    float4 volume;
    float4 attenuation;
    uint4 volumeTextureIndices;
    uint4 volumeSamplerIndices;
    float4 alpha;
    float4 sheen;
    uint4 sheenTextureIndices;
    uint4 sheenSamplerIndices;
    float4 emissive; // rgb: emissiveFactor (incl. strength)
    uint4 emissiveTextureIndices;
    uint4 emissiveSamplerIndices;
    float4 subsurface;            // rgb: diffuseTransmissionColor, a: factor
    uint4 subsurfaceTextureIndices; // x: diffuseTransmissionTexture, y: color texture
    uint4 subsurfaceSamplerIndices;
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] Sampler2D gAlbedo;
[[vk::binding(2, 0)]] Sampler2D gNormal;
[[vk::binding(3, 0)]] Sampler2D gPosition;
[[vk::binding(4, 0)]] StructuredBuffer<GpuLight> gLights;
[[vk::binding(5, 0)]] StructuredBuffer<uint> gClusterCounts;
[[vk::binding(6, 0)]] StructuredBuffer<uint> gClusterIndices;
[[vk::binding(7, 0)]] ConstantBuffer<ShadowUniform> gShadow;
[[vk::binding(8, 0)]] Texture2DArray gShadowMap;
[[vk::binding(9, 0)]] SamplerComparisonState gShadowSampler;
[[vk::binding(10, 0)]] Texture2DArray gSpotShadowMap;
// One six-layer shadow map per point shadow slot (descriptor array).
[[vk::binding(11, 0)]] Texture2DArray gPointShadowMaps[kMaxPointShadowLights];
[[vk::binding(12, 0)]] Texture2D gIndirect;
[[vk::binding(13, 0)]] Texture2D<float> gRtSunShadow;
[[vk::binding(14, 0)]] StructuredBuffer<GpuGiProbe> gGiProbes;
[[vk::binding(15, 0)]] Texture2D gReflections;
[[vk::binding(16, 0)]] Texture2D gSpecularF0;
[[vk::binding(17, 0)]] Texture2D gTransmissionMaterial; // x: transmission, y: ior, z: thickness, w: attenuationDistance
[[vk::binding(18, 0)]] Texture2D gSheenMaterial;        // rgb: sheen color, a: sheen roughness
[[vk::binding(19, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::binding(20, 0)]] Texture2D gMotion; // xy: motion, zw: object UV
[[vk::binding(21, 0)]] Texture2D gBrdfLut; // r: scale, g: bias (split-sum)
[[vk::binding(22, 0)]] SamplerState gBrdfLutSampler;
[[vk::binding(23, 0)]] Texture2D gClouds; // rgb: scattering, a: transmittance
[[vk::binding(24, 0)]] SamplerState gCloudSampler;
[[vk::binding(25, 0)]] Texture2D gGodRays; // rgb: sun-tinted god rays

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];

static const uint kMaxLightsPerCluster = 64;
static const uint kLightTypeDirectional = 0;
static const uint kLightTypePoint = 1;
#include "brdf.slang"
#include "sheen_brdf.slang"
#include "sky.slang"

static const float kShadowMapSize = 1024.0;
static const float kShadowCompareBiasScale = 0.0002;
static const float kPerspectiveCompareBiasScale = 0.001;

static const uint kMaxGiProbes = 64;
static const uint kShadowCascadeCount = 4;
static const uint kMaxSpotShadowLights = 4;
static const uint kSpotShadowOffset = kShadowCascadeCount;
static const uint kPointShadowOffset = kSpotShadowOffset + kMaxSpotShadowLights;
static const uint kPointShadowFaceCount = 6;
static const uint kPointShadowResolution = 512;

float distanceAttenuation(float distance, float range)
{
    const float distanceSquared = max(distance * distance, 1e-4);
    const float ratio = distance / max(range, 1e-4);
    const float ratio2 = ratio * ratio;
    const float ratio4 = ratio2 * ratio2;
    const float cutoff = saturate(1.0 - ratio4);
    return cutoff * cutoff / distanceSquared;
}

float sampleShadowFactor(float3 worldPosition, float3 normal, int2 pixel)
{
    if (gShadow.params.w < 0.5)
    {
        return 1.0;
    }

    if (gShadow.rtParams.x > 0.5)
    {
        // Ray traced sun visibility. The mask is binary per sample, so average a small
        // neighborhood to soften the shadow edges.
        const int2 size = int2(gScene.screenSize.xy);
        float visibility = 0.0;
        [unroll]
        for (int y = -1; y <= 1; ++y)
        {
            [unroll]
            for (int x = -1; x <= 1; ++x)
            {
                const int2 tap = clamp(pixel + int2(x, y), int2(0, 0), size - 1);
                visibility += gRtSunShadow.Load(int3(tap, 0)).r;
            }
        }
        return visibility / 9.0;
    }

    const float viewDepth = -mul(gScene.view, float4(worldPosition, 1.0)).z;
    uint cascade = 0;
    if (viewDepth > gShadow.cascadeSplits.x)
    {
        cascade = 1;
    }
    if (viewDepth > gShadow.cascadeSplits.y)
    {
        cascade = 2;
    }
    if (viewDepth > gShadow.cascadeSplits.z)
    {
        cascade = 3;
    }

    if (viewDepth > gShadow.params.z)
    {
        return 1.0;
    }

    const float3 offsetPosition = worldPosition + normal * gShadow.params.y;
    const float4 lightClip = mul(gShadow.viewProjections[cascade], float4(offsetPosition, 1.0));
    const float3 ndc = lightClip.xyz / max(abs(lightClip.w), 1e-6);
    const float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    const float compareDepth = ndc.z - gShadow.params.x * kShadowCompareBiasScale;
    const float2 texel = float2(1.0 / kShadowMapSize, 1.0 / kShadowMapSize);

    float shadow = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            shadow += gShadowMap.SampleCmpLevelZero(
                gShadowSampler,
                float3(uv + float2(float(x), float(y)) * texel, float(cascade)),
                compareDepth);
        }
    }

    return shadow / 9.0;
}

float sampleSpotShadowFactor(uint shadowIndex, float3 worldPosition, float3 normal)
{
    const uint viewIndex = kSpotShadowOffset + shadowIndex;
    const float3 offsetPosition = worldPosition + normal * gShadow.params.y;
    const float4 lightClip = mul(gShadow.viewProjections[viewIndex], float4(offsetPosition, 1.0));

    if (lightClip.w <= 0.0)
    {
        return 1.0;
    }

    const float3 ndc = lightClip.xyz / lightClip.w;
    const float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    const float compareDepth = ndc.z - gShadow.params.x * kPerspectiveCompareBiasScale;
    const float2 texel = float2(1.0 / kShadowMapSize, 1.0 / kShadowMapSize);

    float shadow = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            shadow += gSpotShadowMap.SampleCmpLevelZero(
                gShadowSampler,
                float3(uv + float2(float(x), float(y)) * texel, float(shadowIndex)),
                compareDepth);
        }
    }

    return shadow / 9.0;
}

// Face order matches the matrix order in ShadowCascade.cpp: +X, -X, +Y, -Y, +Z, -Z.
uint selectPointShadowFace(float3 offset)
{
    const float3 absolute = abs(offset);
    if (absolute.x >= absolute.y && absolute.x >= absolute.z)
    {
        return offset.x > 0.0 ? 0 : 1;
    }
    if (absolute.y >= absolute.z)
    {
        return offset.y > 0.0 ? 2 : 3;
    }
    return offset.z > 0.0 ? 4 : 5;
}

float samplePointShadowFactor(
    uint shadowIndex,
    float3 lightPosition,
    float3 worldPosition,
    float3 normal)
{
    const float3 offsetPosition = worldPosition + normal * gShadow.params.y;
    const uint face = selectPointShadowFace(offsetPosition - lightPosition);
    const uint viewIndex = kPointShadowOffset + shadowIndex * kPointShadowFaceCount + face;

    const float4 lightClip = mul(gShadow.viewProjections[viewIndex], float4(offsetPosition, 1.0));
    if (lightClip.w <= 0.0)
    {
        return 1.0;
    }

    const float3 ndc = lightClip.xyz / lightClip.w;
    const float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }

    // Each point light owns its own map; the cube face is selected by the array layer.
    const float2 texel = 1.0 / float(kPointShadowResolution);
    const float compareDepth = ndc.z - gShadow.params.x * kPerspectiveCompareBiasScale;

    float shadow = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            // Clamp the PCF taps inside the face so neighbouring cube faces never bleed in.
            const float2 tapUv = clamp(
                uv + float2(float(x), float(y)) * texel,
                float2(0.0, 0.0),
                float2(1.0, 1.0));
            shadow += gPointShadowMaps[NonUniformResourceIndex(shadowIndex)]
                          .SampleCmpLevelZero(gShadowSampler, float3(tapUv, float(face)), compareDepth);
        }
    }

    return shadow / 9.0;
}

uint clusterIndexOf(float3 worldPosition, float2 pixelCoord)
{
    const float3 viewPosition = mul(gScene.view, float4(worldPosition, 1.0)).xyz;
    const float viewDepth = max(-viewPosition.z, gScene.clusterParams.x);
    const float slice = log(viewDepth / gScene.clusterParams.x) /
                        log(gScene.clusterParams.y / gScene.clusterParams.x) * gScene.clusterParams.w;

    const uint x = min(uint(pixelCoord.x / gScene.clusterParams.z), gScene.clusterGrid.x - 1);
    const uint y = min(uint(pixelCoord.y / gScene.clusterParams.z), gScene.clusterGrid.y - 1);
    const uint z = min(uint(slice), gScene.clusterGrid.z - 1);
    return (z * gScene.clusterGrid.y + y) * gScene.clusterGrid.x + x;
}

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

// The lighting pass renders offscreen with two attachments: the HDR lit image and the subsurface
// diffuse source (rgb: transmit energy, a: coverage). The composite pass adds the diffused source
// and applies tone mapping, so this entry no longer tonemaps.
struct LightingOutput
{
    float4 color : SV_Target0;
    float4 subsurface : SV_Target1;
};

LightingOutput MakeOutput(float3 color, float3 subsurface, float subsurfaceCoverage)
{
    LightingOutput output;
    output.color = float4(color, 1.0);
    output.subsurface = float4(subsurface, subsurfaceCoverage);
    return output;
}

[shader("fragment")]
LightingOutput fragmentMain(VertexOutput input)
{
    const float4 albedoMetalness = gAlbedo.Sample(input.uv);
    const float4 normalRoughness = gNormal.Sample(input.uv);
    const float4 worldPosition = gPosition.Sample(input.uv);

    if (worldPosition.w < 0.5)
    {
        float3 sky = gScene.ambientColor.rgb;
        if (gScene.skyParams.x > 0.5)
        {
            const float2 ndc = input.uv * 2.0 - 1.0;
            const float4 clip = mul(gScene.inverseViewProjection, float4(ndc, 1.0, 1.0));
            const float3 direction =
                normalize(clip.xyz / max(clip.w, 1e-6) - gScene.cameraPosition.xyz);
            sky = EnvironmentBackground(
                direction,
                gScene.skyZenith.rgb,
                gScene.skyHorizon.rgb,
                gScene.skyGround.rgb,
                gScene.skyParams.y,
                gScene.sunDirection.xyz,
                gScene.sunColor.rgb,
                gScene.skyParams.z,
                uint(gScene.skyMap.x + 0.5),
                uint(gScene.skyMap.y + 0.5),
                gScene.skyMap.z > 0.5);
        }

        // Nubis-style clouds composite over the sky: coverage attenuates the background and adds
        // the in-scattered light.
        const float4 clouds = gClouds.Sample(gCloudSampler, input.uv);
        sky = sky * clouds.a + clouds.rgb + gGodRays.Sample(gCloudSampler, input.uv).rgb;
        return MakeOutput(sky, float3(0.0), 0.0);
    }

    const float3 albedo = albedoMetalness.rgb;
    const float metalness = albedoMetalness.a;
    const float3 normal = normalize(normalRoughness.xyz * 2.0 - 1.0);
    const float roughness = clamp(normalRoughness.a, 0.04, 1.0);
    const float3 viewDirection = normalize(gScene.cameraPosition.xyz - worldPosition.xyz);
    const float3 dielectricF0 = gSpecularF0.Load(int3(int2(input.position.xy), 0)).rgb;
    const float3 f0 = lerp(dielectricF0, albedo, metalness);
    // Transmissive surfaces keep only the (1 - transmission) part of the surface response; the
    // traced specular signal carries the reflection + refraction lobes.
    const float transmission = saturate(gTransmissionMaterial.Load(int3(int2(input.position.xy), 0)).x);
    const float surfaceWeight = 1.0 - transmission;
    const float4 sheenMaterial = gSheenMaterial.Load(int3(int2(input.position.xy), 0));
    const float3 sheenColor = sheenMaterial.rgb;
    const float sheenRoughness = sheenMaterial.a;

    // Emissive: position.w stores the object index + 1 (0 = sky) and motion.zw the object UV, so
    // the emissive texture is sampled with its real texture coordinates instead of screen UVs.
    // The same fetch drives subsurface scattering (KHR_materials_diffuse_transmission): the
    // transmitted diffuse energy moves out of the diffuse BRDF into the back-lit BTDF.
    const float4 gbufferMotion = gMotion.Load(int3(int2(input.position.xy), 0));
    float3 emissive = 0.0;
    float3 subsurfaceColor = float3(0.0);
    float subsurfaceFactor = 0.0;
    if (worldPosition.w > 0.5)
    {
        const GpuObjectData object = gObjects[uint(worldPosition.w - 0.5)];
        const float3 emissiveSample =
            gTextures[NonUniformResourceIndex(object.emissiveTextureIndices.x)].Sample(
                gSamplers[NonUniformResourceIndex(object.emissiveSamplerIndices.x)],
                gbufferMotion.zw).rgb;
        emissive = object.emissive.rgb * emissiveSample;

        const float subsurfaceFactorSample =
            gTextures[NonUniformResourceIndex(object.subsurfaceTextureIndices.x)].Sample(
                gSamplers[NonUniformResourceIndex(object.subsurfaceSamplerIndices.x)],
                gbufferMotion.zw).a;
        const float3 subsurfaceColorSample =
            gTextures[NonUniformResourceIndex(object.subsurfaceTextureIndices.y)].Sample(
                gSamplers[NonUniformResourceIndex(object.subsurfaceSamplerIndices.y)],
                gbufferMotion.zw).rgb;
        subsurfaceFactor = object.subsurface.a * subsurfaceFactorSample;
        subsurfaceColor = object.subsurface.rgb * subsurfaceColorSample;
    }
    const float3 diffuseAlbedo = albedo * (1.0 - subsurfaceFactor);

    // Ambient/indirect diffuse: SH irradiance from the environment, or the flat ambient fallback.
    // `color` accumulates the indirect signal; direct light goes into `directColor` so the GI-only
    // debug view can show the indirect part alone.
    float3 color = gScene.skyParams.x > 0.5
                       ? EvaluateSkySh(gScene.skySh, normal) * diffuseAlbedo * (1.0 - metalness) * surfaceWeight
                       : gScene.ambientColor.rgb * diffuseAlbedo * surfaceWeight;
    float3 directColor = 0.0;
    // Subsurface diffuse transmission is written to the second attachment and diffused by the
    // screen space passes; the composite adds the blurred result.
    float3 subsurfaceEnergy = float3(0.0);
    const float subsurfaceCoverage = subsurfaceFactor > 0.0 ? 1.0 : 0.0;

    const float shadowFactor = sampleShadowFactor(worldPosition.xyz, normal, int2(input.position.xy));

    const uint clusterIndex = clusterIndexOf(worldPosition.xyz, input.position.xy);
    const uint clusterBase = clusterIndex * kMaxLightsPerCluster;
    const uint clusterLightCount = min(gClusterCounts[clusterIndex], kMaxLightsPerCluster);

    if (gScene.screenSize.z > 0.5 && gScene.screenSize.z < 1.5)
    {
        const uint rawCount = gClusterCounts[clusterIndex];
        if (rawCount >= kMaxLightsPerCluster)
        {
            return MakeOutput(float3(1.0, 0.0, 1.0), float3(0.0), 0.0);
        }

        const float heat = saturate(float(rawCount) / 64.0);
        return MakeOutput(float3(heat, heat * heat, 1.0 - heat), float3(0.0), 0.0);
    }

    for (uint slot = 0; slot < clusterLightCount; ++slot)
    {
        const uint lightIndex = gClusterIndices[clusterBase + slot];
        if (lightIndex >= gScene.clusterGrid.w)
        {
            continue;
        }

        const GpuLight light = gLights[lightIndex];
        const uint type = uint(light.params.x);

        float3 lightDirection = float3(0.0, 0.0, 0.0);
        float3 radiance = light.color.rgb * light.directionIntensity.w;
        if (type == kLightTypeDirectional)
        {
            lightDirection = -light.directionIntensity.xyz;
            radiance *= shadowFactor;
        }
        else
        {
            const float3 offset = light.positionRange.xyz - worldPosition.xyz;
            const float distance = length(offset);
            lightDirection = offset / max(distance, 1e-4);
            radiance *= distanceAttenuation(distance, light.positionRange.w);

            if (type != kLightTypePoint)
            {
                const float spotCosine = dot(-lightDirection, light.directionIntensity.xyz);
                const float spotFalloff = saturate(
                    (spotCosine - light.params.z) / max(light.params.y - light.params.z, 1e-4));
                radiance *= spotFalloff * spotFalloff;
            }
        }

        const uint shadowSlot = uint(light.params.w + 0.5);
        if (shadowSlot > 0)
        {
            const uint shadowIndex = shadowSlot - 1;
            if (type == kLightTypePoint)
            {
                radiance *= samplePointShadowFactor(
                    shadowIndex,
                    light.positionRange.xyz,
                    worldPosition.xyz,
                    normal);
            }
            else if (type != kLightTypeDirectional)
            {
                radiance *= sampleSpotShadowFactor(shadowIndex, worldPosition.xyz, normal);
            }
        }

        const float rawNdotL = dot(normal, lightDirection);
        const float nDotL = max(rawNdotL, 0.0);
        if (nDotL > 0.0)
        {
            directColor += EvaluateCookTorrance(
                               diffuseAlbedo,
                               f0,
                               metalness,
                               roughness,
                               normal,
                               viewDirection,
                               lightDirection,
                               radiance) *
                           surfaceWeight;
            directColor += EvaluateSheenBRDF(
                               sheenColor,
                               sheenRoughness,
                               normal,
                               viewDirection,
                               lightDirection,
                               radiance,
                               nDotL) *
                           surfaceWeight;
        }
        if (subsurfaceFactor > 0.0)
        {
            // Back-lit diffuse transmission (KHR_materials_diffuse_transmission). The screen space
            // diffusion pass blurs this contribution over the surface.
            subsurfaceEnergy += SubsurfaceTransmissionRadiance(
                                    subsurfaceColor,
                                    subsurfaceFactor,
                                    metalness,
                                    rawNdotL,
                                    radiance) *
                                surfaceWeight;
        }
    }

    if (gScene.skyParams.x > 0.5)
    {
        // Image based specular: prefiltered environment times the split-sum BRDF terms.
        const float3 reflection = reflect(-viewDirection, normal);
        const float3 prefiltered = EnvironmentPrefiltered(
            reflection,
            roughness,
            gScene.skyZenith.rgb,
            gScene.skyHorizon.rgb,
            gScene.skyGround.rgb,
            gScene.skyParams.y,
            gScene.skySh,
            uint(gScene.skyMap.x + 0.5),
            uint(gScene.skyMap.y + 0.5),
            gScene.skyMap.z > 0.5);
        const float2 brdf = gBrdfLut
                                .Sample(gBrdfLutSampler, float2(saturate(dot(normal, viewDirection)), roughness))
                                .rg;
        color += prefiltered * (f0 * brdf.x + brdf.y) * surfaceWeight;
    }

    if (gScene.screenSize.w > 0.5)
    {
        color += gIndirect.Load(int3(int2(input.position.xy), 0)).rgb * surfaceWeight;
    }
    else if (gScene.ambientColor.w > 0.5)
    {
        // Probe GI: inverse distance blend of the surrounding irradiance probes with a depth
        // visibility term so probes on the far side of geometry do not bleed light through walls.
        const uint probeCount = min(uint(gScene.ambientColor.w + 0.5), kMaxGiProbes);
        float3 indirect = 0.0;
        float weightSum = 0.0;
        for (uint probeIndex = 0; probeIndex < probeCount; ++probeIndex)
        {
            const GpuGiProbe probe = gGiProbes[probeIndex];
            if (probe.depth.z < 0.5)
            {
                // Probe failed the stuck-in-geometry depth check.
                continue;
            }

            const float3 offset = probe.position.xyz - worldPosition.xyz;
            const float distanceSquared = dot(offset, offset);
            const float distance = sqrt(distanceSquared);

            // Chebyshev visibility (DDGI): probes whose average hit distance is much smaller than
            // the shading distance are occluded by geometry between the probe and the surface.
            const float mean = max(probe.depth.x, 1e-4);
            const float meanSquared = max(probe.depth.y, mean * mean + 1e-4);
            const float variance = max(meanSquared - mean * mean, 1e-4);
            const float delta = distance - mean;
            const float visibility =
                delta > 0.0 ? variance / (variance + delta * delta) : 1.0;

            const float weight = visibility / (distanceSquared + 0.05);
            indirect +=
                (probe.sh0.rgb + probe.shX.rgb * normal.x + probe.shY.rgb * normal.y +
                 probe.shZ.rgb * normal.z) * weight;
            weightSum += weight;
        }
        if (weightSum > 0.0)
        {
            color += albedo * indirect / weightSum * surfaceWeight;
        }
    }

    if (gScene.reflectionParams.x > 0.0)
    {
        // Ray traced glossy reflections (already fresnel weighted) at the GI trace resolution.
        color += gReflections.Load(int3(int2(input.position.xy), 0)).rgb *
                 gScene.reflectionParams.x;
    }

    // GI-only debug view: the indirect signal (ambient SH/IBL diffuse, traced indirect diffuse or
    // probes and reflections) without direct lights or emissive. The composite applies the tone
    // mapping (the renderer disables the subsurface add for this view).
    if (gScene.screenSize.z > 1.5)
    {
        return MakeOutput(color, float3(0.0), 0.0);
    }

    return MakeOutput(color + directColor + emissive, subsurfaceEnergy, subsurfaceCoverage);
}
