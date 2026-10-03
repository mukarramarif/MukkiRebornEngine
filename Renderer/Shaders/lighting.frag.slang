struct SceneUniform
{
    float4x4 viewProjection;
    float4x4 previousViewProjection;
    float4x4 view;
    float4 cameraPosition;
    float4 ambientColor;
    float4 clusterParams; // x: near, y: far, z: tile size in pixels, w: z slice count
    uint4 clusterGrid;    // x, y, z grid dimensions, w: light count
    float4 screenSize;
    float4 reflectionParams; // x: reflection intensity
    float4 sunDirection;     // xyz: direction toward the sun
    float4 sunColor;         // rgb: color * intensity
};

struct GpuLight
{
    float4 positionRange;
    float4 directionIntensity;
    float4 color;
    float4 params; // x: type, y: inner cone cosine, z: outer cone cosine
};

struct ShadowUniform
{
    float4x4 viewProjections[20]; // 4 cascades + 4 spots + 2 point lights * 6 faces
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
[[vk::binding(11, 0)]] Texture2D gPointShadowMap;
[[vk::binding(12, 0)]] Texture2D gIndirect;
[[vk::binding(13, 0)]] Texture2D<float> gRtSunShadow;
[[vk::binding(14, 0)]] StructuredBuffer<GpuGiProbe> gGiProbes;
[[vk::binding(15, 0)]] Texture2D gReflections;
[[vk::binding(16, 0)]] Texture2D gSpecularF0;
[[vk::binding(17, 0)]] Texture2D gTransmissionMaterial; // x: transmission, y: ior, z: thickness, w: attenuationDistance
[[vk::binding(18, 0)]] Texture2D gSheenMaterial;        // rgb: sheen color, a: sheen roughness

static const uint kMaxLightsPerCluster = 64;
static const uint kLightTypeDirectional = 0;
static const uint kLightTypePoint = 1;
#include "sheen_brdf.slang"

static const float kPi = 3.14159265359;
static const float kShadowMapSize = 1024.0;
static const float kShadowCompareBiasScale = 0.0002;
static const float kPerspectiveCompareBiasScale = 0.001;

static const uint kMaxGiProbes = 64;
static const uint kShadowCascadeCount = 4;
static const uint kMaxSpotShadowLights = 4;
static const uint kSpotShadowOffset = kShadowCascadeCount;
static const uint kPointShadowOffset = kSpotShadowOffset + kMaxSpotShadowLights;
static const uint kPointShadowFaceCount = 6;
static const float kPointShadowTileSize = 512.0;
static const float kPointShadowAtlasColumns = 3.0;
static const float kPointShadowAtlasWidth = 1536.0;  // 3 tiles wide
static const float kPointShadowAtlasHeight = 2048.0; // 2 point lights * 2 tile rows

float3 acesTonemap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

float3 fresnelSchlick(float cosTheta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float distributionGgx(float nDotH, float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(kPi * denominator * denominator, 1e-6);
}

float geometrySchlickGgx(float nDotX, float roughness)
{
    const float r = roughness + 1.0;
    const float k = (r * r) / 8.0;
    return nDotX / max(nDotX * (1.0 - k) + k, 1e-4);
}

float3 evaluateBrdf(
    float3 albedo,
    float3 f0,
    float metalness,
    float roughness,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    float3 radiance,
    float nDotL)
{
    const float3 halfVector = normalize(viewDirection + lightDirection);
    const float nDotV = max(dot(normal, viewDirection), 1e-4);
    const float nDotH = max(dot(normal, halfVector), 0.0);
    const float vDotH = max(dot(viewDirection, halfVector), 0.0);

    const float3 fresnel = fresnelSchlick(vDotH, f0);
    const float distribution = distributionGgx(nDotH, roughness);
    const float geometry = geometrySchlickGgx(nDotV, roughness) * geometrySchlickGgx(nDotL, roughness);
    const float3 specular = (distribution * geometry * fresnel) / max(4.0 * nDotV * nDotL, 1e-4);

    const float3 diffuseWeight = (1.0 - fresnel) * (1.0 - metalness);
    return (diffuseWeight * albedo / kPi + specular) * radiance * nDotL;
}

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

    // Map the face UV into its atlas tile and clamp each PCF tap inside the tile so neighbouring
    // faces never bleed into the filter.
    const uint tileColumn = face % uint(kPointShadowAtlasColumns);
    const uint tileRow = shadowIndex * 2u + face / uint(kPointShadowAtlasColumns);
    const float2 atlasSize = float2(kPointShadowAtlasWidth, kPointShadowAtlasHeight);
    const float2 tileOrigin = float2(float(tileColumn), float(tileRow)) * kPointShadowTileSize;
    const float2 tileMinUv = (tileOrigin + 0.5) / atlasSize;
    const float2 tileMaxUv = (tileOrigin + kPointShadowTileSize - 0.5) / atlasSize;
    const float2 atlasUv = (tileOrigin + uv * kPointShadowTileSize) / atlasSize;
    const float2 texel = 1.0 / atlasSize;

    const float compareDepth = ndc.z - gShadow.params.x * kPerspectiveCompareBiasScale;

    float shadow = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float2 tapUv = clamp(
                atlasUv + float2(float(x), float(y)) * texel,
                tileMinUv,
                tileMaxUv);
            shadow += gPointShadowMap.SampleCmpLevelZero(gShadowSampler, tapUv, compareDepth);
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

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    const float4 albedoMetalness = gAlbedo.Sample(input.uv);
    const float4 normalRoughness = gNormal.Sample(input.uv);
    const float4 worldPosition = gPosition.Sample(input.uv);

    if (worldPosition.w < 0.5)
    {
        return float4(gScene.ambientColor.rgb, 1.0);
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

    float3 color = gScene.ambientColor.rgb * albedo * surfaceWeight;

    const float shadowFactor = sampleShadowFactor(worldPosition.xyz, normal, int2(input.position.xy));

    const uint clusterIndex = clusterIndexOf(worldPosition.xyz, input.position.xy);
    const uint clusterBase = clusterIndex * kMaxLightsPerCluster;
    const uint clusterLightCount = min(gClusterCounts[clusterIndex], kMaxLightsPerCluster);

    if (gScene.screenSize.z > 0.5)
    {
        const uint rawCount = gClusterCounts[clusterIndex];
        if (rawCount >= kMaxLightsPerCluster)
        {
            return float4(1.0, 0.0, 1.0, 1.0);
        }

        const float heat = saturate(float(rawCount) / 64.0);
        return float4(heat, heat * heat, 1.0 - heat, 1.0);
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

        const float nDotL = max(dot(normal, lightDirection), 0.0);
        if (nDotL > 0.0)
        {
            color += evaluateBrdf(
                         albedo,
                         f0,
                         metalness,
                         roughness,
                         normal,
                         viewDirection,
                         lightDirection,
                         radiance,
                         nDotL) *
                     surfaceWeight;
            color += EvaluateSheenBRDF(
                         sheenColor,
                         sheenRoughness,
                         normal,
                         viewDirection,
                         lightDirection,
                         radiance,
                         nDotL) *
                     surfaceWeight;
        }
    }

    if (gScene.screenSize.w > 0.5)
    {
        color += gIndirect.Load(int3(int2(input.position.xy), 0)).rgb * surfaceWeight;
    }
    else if (gScene.ambientColor.w > 0.5)
    {
        // Probe GI: inverse distance blend of the surrounding irradiance probes.
        const uint probeCount = min(uint(gScene.ambientColor.w + 0.5), kMaxGiProbes);
        float3 indirect = 0.0;
        float weightSum = 0.0;
        for (uint probeIndex = 0; probeIndex < probeCount; ++probeIndex)
        {
            const GpuGiProbe probe = gGiProbes[probeIndex];
            const float3 offset = probe.position.xyz - worldPosition.xyz;
            const float weight = 1.0 / (dot(offset, offset) + 0.05);
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

    return float4(acesTonemap(color), 1.0);
}
