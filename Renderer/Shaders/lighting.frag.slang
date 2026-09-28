struct SceneUniform
{
    float4x4 viewProjection;
    float4x4 view;
    float4 cameraPosition;
    float4 ambientColor;
    float4 clusterParams; // x: near, y: far, z: tile size in pixels, w: z slice count
    uint4 clusterGrid;    // x, y, z grid dimensions, w: light count
    float4 screenSize;    // x: width, y: height
};

struct GpuLight
{
    float4 positionRange;
    float4 directionIntensity;
    float4 color;
    float4 params; // x: type, y: inner cone cosine, z: outer cone cosine
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] Sampler2D gAlbedo;
[[vk::binding(2, 0)]] Sampler2D gNormal;
[[vk::binding(3, 0)]] Sampler2D gPosition;
[[vk::binding(4, 0)]] StructuredBuffer<GpuLight> gLights;
[[vk::binding(5, 0)]] StructuredBuffer<uint> gClusterCounts;
[[vk::binding(6, 0)]] StructuredBuffer<uint> gClusterIndices;

static const uint kMaxLightsPerCluster = 64;
static const uint kLightTypeDirectional = 0;
static const uint kLightTypePoint = 1;
static const float kPi = 3.14159265359;

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

    const float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metalness);
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

    float3 color = gScene.ambientColor.rgb * albedo;

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

        const float nDotL = max(dot(normal, lightDirection), 0.0);
        if (nDotL > 0.0)
        {
            color += evaluateBrdf(
                albedo,
                metalness,
                roughness,
                normal,
                viewDirection,
                lightDirection,
                radiance,
                nDotL);
        }
    }

    return float4(acesTonemap(color), 1.0);
}
