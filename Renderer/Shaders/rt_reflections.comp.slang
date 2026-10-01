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
    float4 params5; // x: probeRays, y: probeHysteresis, z: probeCount, w: reflectionResetHistory
    float4 params6; // x: reflectionMaxDistance, y: reflectionMaxRoughness, z: reflectionSamples, w: reflectionIntensity
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
[[vk::binding(5, 0)]] Texture2D gFeedback;
[[vk::binding(6, 0)]] RWTexture2D<float4> gReflectionOut;
[[vk::binding(7, 0)]] RWTexture2D<float> gHitDistanceOut;
[[vk::binding(8, 0)]] Texture2D gAlbedo;
[[vk::binding(9, 0)]] Texture2D<float> gRtSunShadow;
[[vk::binding(10, 0)]] Texture2D gSpecularF0;

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];

#include "gi_shading.slang"

// Traces glossy reflection rays at the GI trace resolution. Hit radiance is shaded with the shared
// path tracer shading and weighted by a Schlick fresnel; smooth surfaces get sharp reflections and
// the result fades out above the roughness cutoff.
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
    const float4 normalRoughness = gNormal.Load(int3(fullPixel, 0));
    const float roughness = normalRoughness.w;
    if (depth <= 0.0 || depth >= 1.0 || roughness > gGi.params6.y)
    {
        gReflectionOut[halfPixel] = float4(0.0, 0.0, 0.0, 1.0);
        gHitDistanceOut[halfPixel] = 0.0;
        return;
    }

    const float3 normal = normalize(normalRoughness.xyz * 2.0 - 1.0);
    const float4 albedoMetalness = gAlbedo.Load(int3(fullPixel, 0));
    const float3 dielectricF0 = gSpecularF0.Load(int3(fullPixel, 0)).rgb;
    const float3 f0 = lerp(dielectricF0, albedoMetalness.rgb, albedoMetalness.a);

    const float2 ndc = (float2(fullPixel) + 0.5) / gGi.screenSize.xy * 2.0 - 1.0;
    const float4 worldClip = mul(gGi.inverseViewProjection, float4(ndc, depth, 1.0));
    const float3 worldPosition = worldClip.xyz / worldClip.w;
    const float3 viewDirection = normalize(gGi.cameraPosition.xyz - worldPosition);
    const float3 origin = worldPosition + normal * 0.01;

    const uint frame = uint(gGi.params.y);
    const uint sampleCount = uint(clamp(gGi.params6.z, 1.0, 8.0));

    float3 radianceAccum = 0.0;
    float hitDistanceAccum = 0.0;

    for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        const uint sampleSeed = frame + sampleIndex * 4099u;
        const float3 direction =
            SampleGgxDirection(normal, viewDirection, roughness, dispatchThreadId.xy, sampleSeed);

        RayQuery<RAY_FLAG_NONE> query;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = 0.0;
        ray.TMax = gGi.params6.x;
        query.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
        query.Proceed();

        float3 radiance = gGi.skyColor.rgb * gGi.skyColor.w;
        float hitDistance = gGi.params6.x;
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            hitDistance = query.CommittedRayT();
            radiance = ShadeHit(
                gFeedback,
                query.CommittedInstanceID(),
                query.CommittedPrimitiveIndex(),
                query.CommittedTriangleBarycentrics(),
                origin,
                direction,
                hitDistance,
                dispatchThreadId.xy,
                sampleSeed);
        }

        const float vDotR = saturate(dot(viewDirection, direction));
        const float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vDotR, 5.0);

        radianceAccum += radiance * fresnel;
        hitDistanceAccum += hitDistance;
    }

    const float inverseSamples = 1.0 / float(sampleCount);
    gReflectionOut[halfPixel] =
        float4(clamp(radianceAccum * inverseSamples, 0.0, kMaxGiRadiance), 1.0);
    gHitDistanceOut[halfPixel] = hitDistanceAccum * inverseSamples;
}
