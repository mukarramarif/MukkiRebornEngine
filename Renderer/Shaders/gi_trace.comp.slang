struct GiUniform
{
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    float4x4 viewProjection;
    float4 screenSize;
    float4 halfSize;
    float4 cameraPosition;
    float4 skyColor;              // rgb: color, w: intensity
    float4 sunDirectionIntensity; // xyz: direction toward the light, w: intensity
    float4 sunColor;
    float4 params;  // x: maxDistance, y: frameCounter, z: specularMinRoughness, w: feedbackIntensity
    float4 params2; // x: temporalAlphaMin, y: phiDepth, z: phiNormal, w: phiLuminance
    float4 params3; // x: phiHitDistance, y: intensity, z: sunShadowRay, w: resetHistory
    float4 params4; // x: sunShadowSamples, y: sunAngularRadius (radians), z: traceSamples, w: rtSunMask
    float4 params5; // x: probeRays, y: probeHysteresis, z: probeCount, w: unused
};

struct GpuGeometryData
{
    float4 baseColorFactor;
    uint4 bufferIndices;  // x: vertex buffer, y: index buffer, z: vertex stride
    uint4 textureIndices; // x: base color, y: metallic-roughness
    uint4 samplerIndices; // x: base color sampler, y: metallic-roughness sampler
    float4 params;        // x: roughness, y: metalness
};

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] RaytracingAccelerationStructure gTlas;
[[vk::binding(2, 0)]] StructuredBuffer<GpuGeometryData> gGeometry;
[[vk::binding(3, 0)]] Texture2D gDepth;
[[vk::binding(4, 0)]] Texture2D gNormal;
[[vk::binding(5, 0)]] Texture2D gIndirect;
[[vk::binding(6, 0)]] RWTexture2D<float4> gDiffuseOut;
[[vk::binding(7, 0)]] RWTexture2D<float4> gSpecularOut;
[[vk::binding(8, 0)]] RWTexture2D<float4> gHitDistanceOut;
[[vk::binding(9, 0)]] Texture2D<float> gRtSunShadow;

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];

#include "gi_shading.slang"

float3 SampleGgxDirection(float3 normal, float3 viewDirection, float roughness, uint2 pixel, uint frame)
{
    const float3 view = normalize(viewDirection);
    const float3 tangent =
        normalize(cross(abs(view.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), view));
    const float3 bitangent = cross(view, tangent);

    const float a = max(roughness * roughness, 1e-4);
    const float r1 = Hash01(pixel, frame, 3u);
    const float r2 = Hash01(pixel, frame, 7u);
    const float phi = 2.0 * kPi * r1;
    const float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    const float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    const float3 halfVector =
        tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + view * cosTheta;
    return reflect(-view, normalize(halfVector));
}

[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 halfPixel = int2(dispatchThreadId.xy);
    if (halfPixel.x >= int(gGi.halfSize.x) || halfPixel.y >= int(gGi.halfSize.y))
    {
        return;
    }

    const float2 uv = (float2(halfPixel) + 0.5) / gGi.halfSize.xy;
    const int2 fullPixel = clamp(int2(uv * gGi.screenSize.xy), int2(0, 0), int2(gGi.screenSize.xy) - 1);

    const float depth = gDepth.Load(int3(fullPixel, 0)).r;
    if (depth <= 0.0 || depth >= 1.0)
    {
        gDiffuseOut[halfPixel] = float4(0.0, 0.0, 0.0, 1.0);
        gSpecularOut[halfPixel] = float4(0.0, 0.0, 0.0, 1.0);
        gHitDistanceOut[halfPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float4 normalRoughness = gNormal.Load(int3(fullPixel, 0));
    const float3 normal = normalize(normalRoughness.xyz * 2.0 - 1.0);
    const float roughness = clamp(normalRoughness.w, gGi.params.z, 1.0);

    const float2 ndc = (float2(fullPixel) + 0.5) / gGi.screenSize.xy * 2.0 - 1.0;
    const float4 worldClip = mul(gGi.inverseViewProjection, float4(ndc, depth, 1.0));
    const float3 worldPosition = worldClip.xyz / worldClip.w;
    const float3 viewDirection = normalize(gGi.cameraPosition.xyz - worldPosition);

    const uint frame = uint(gGi.params.y);
    const uint traceSamples = uint(clamp(gGi.params4.z, 1.0, 8.0));
    const float3 origin = worldPosition + normal * 0.01;
    const float3 f0 = float3(0.04, 0.04, 0.04);

    float3 diffuseAccum = 0.0;
    float3 specularAccum = 0.0;
    float hitDistanceAccum = 0.0;

    for (uint sampleIndex = 0; sampleIndex < traceSamples; ++sampleIndex)
    {
        // Offset the hash seed per sample so each path uses its own noise pattern.
        const uint sampleSeed = frame + sampleIndex * 4099u;

        const float3 diffuseDirection = SampleCosineDirection(normal, dispatchThreadId.xy, sampleSeed);
        RayQuery<RAY_FLAG_NONE> diffuseQuery;
        RayDesc diffuseRay;
        diffuseRay.Origin = origin;
        diffuseRay.Direction = diffuseDirection;
        diffuseRay.TMin = 0.0;
        diffuseRay.TMax = gGi.params.x;
        diffuseQuery.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, diffuseRay);
        diffuseQuery.Proceed();

        float3 diffuseRadiance = gGi.skyColor.rgb * gGi.skyColor.w;
        float diffuseHitDistance = gGi.params.x;
        if (diffuseQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            diffuseHitDistance = diffuseQuery.CommittedRayT();
            diffuseRadiance = ShadeHit(
                gIndirect,
                diffuseQuery.CommittedInstanceID(),
                diffuseQuery.CommittedPrimitiveIndex(),
                diffuseQuery.CommittedTriangleBarycentrics(),
                origin,
                diffuseDirection,
                diffuseHitDistance,
                dispatchThreadId.xy,
                sampleSeed);
        }

        const float3 specularDirection =
            SampleGgxDirection(normal, viewDirection, roughness, dispatchThreadId.xy, sampleSeed + 7919u);
        RayQuery<RAY_FLAG_NONE> specularQuery;
        RayDesc specularRay;
        specularRay.Origin = origin;
        specularRay.Direction = specularDirection;
        specularRay.TMin = 0.0;
        specularRay.TMax = gGi.params.x;
        specularQuery.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, specularRay);
        specularQuery.Proceed();

        float3 specularRadiance = gGi.skyColor.rgb * gGi.skyColor.w;
        if (specularQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            specularRadiance = ShadeHit(
                gIndirect,
                specularQuery.CommittedInstanceID(),
                specularQuery.CommittedPrimitiveIndex(),
                specularQuery.CommittedTriangleBarycentrics(),
                origin,
                specularDirection,
                specularQuery.CommittedRayT(),
                dispatchThreadId.xy,
                sampleSeed);
        }

        const float nDotL = max(dot(normal, specularDirection), 0.0);
        const float vDotH = max(dot(viewDirection, normalize(viewDirection + specularDirection)), 0.0);
        const float3 fresnel = f0 + (1.0 - f0) * pow(saturate(1.0 - vDotH), 5.0);

        diffuseAccum += diffuseRadiance;
        specularAccum += specularRadiance * fresnel * nDotL;
        hitDistanceAccum += diffuseHitDistance;
    }

    const float inverseSamples = 1.0 / float(traceSamples);
    gDiffuseOut[halfPixel] = float4(diffuseAccum * inverseSamples, 1.0);
    gSpecularOut[halfPixel] = float4(specularAccum * inverseSamples, 1.0);
    gHitDistanceOut[halfPixel] = float4(hitDistanceAccum * inverseSamples, 0.0, 0.0, 1.0);
}
