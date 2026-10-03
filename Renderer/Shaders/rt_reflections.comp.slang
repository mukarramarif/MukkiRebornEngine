#include "gi_structs.slang"

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
[[vk::binding(11, 0)]] Texture2D gTransmission;
[[vk::binding(12, 0)]] Texture2D gAttenuation;

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];

#include "gi_shading.slang"

float3 TraceSpecularRay(
    float3 origin,
    float3 direction,
    uint2 pixel,
    uint seed,
    out float hitDistance)
{
    RayQuery<RAY_FLAG_NONE> query;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = gGi.params6.x;
    query.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    hitDistance = gGi.params6.x;
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        return gGi.skyColor.rgb * gGi.skyColor.w;
    }

    hitDistance = query.CommittedRayT();
    return ShadeHit(
        gFeedback,
        query.CommittedInstanceID(),
        query.CommittedPrimitiveIndex(),
        query.CommittedTriangleBarycentrics(),
        origin,
        direction,
        hitDistance,
        pixel,
        seed);
}

// Refracts `incident` around the microfacet normal `halfVector`.
float3 RefractAroundMicrofacet(float3 incident, float3 halfVector, float ior, out bool totalInternal)
{
    float3 h = halfVector;
    if (dot(h, incident) < 0.0)
    {
        h = -h;
    }

    const float eta = 1.0 / max(ior, 1.0);
    const float cosI = saturate(dot(incident, h));
    const float k = 1.0 - eta * eta * (1.0 - cosI * cosI);
    totalInternal = k < 0.0;
    if (totalInternal)
    {
        return reflect(incident, h);
    }

    const float3 refracted = eta * incident + (eta * cosI - sqrt(k)) * h;
    const float refractedLengthSquared = dot(refracted, refracted);
    if (refractedLengthSquared <= 1e-8)
    {
        totalInternal = true;
        return reflect(incident, h);
    }
    return refracted * rsqrt(refractedLengthSquared);
}

// Traces glossy specular rays at the GI trace resolution. Opaque surfaces trace GGX reflection
// rays; transmissive surfaces split reflect/refract by fresnel using the same microfacet normal,
// including volume absorption and optional chromatic dispersion.
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

    const float4 transmissionData = gTransmission.Load(int3(fullPixel, 0));
    const float transmission = saturate(transmissionData.x);
    const float ior = max(transmissionData.y, 1.0);
    const float thickness = max(transmissionData.z, 0.0);
    const float attenuationDistance = max(transmissionData.w, 0.0);
    const float4 attenuationData = gAttenuation.Load(int3(fullPixel, 0));
    const float3 attenuationColor = max(attenuationData.rgb, float3(1e-3, 1e-3, 1e-3));
    const float dispersion = max(attenuationData.a, 0.0);

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
        const float3 halfVector =
            SampleGgxHalfVector(viewDirection, roughness, dispatchThreadId.xy, sampleSeed);
        const float vDotH = saturate(dot(viewDirection, halfVector));
        const float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
        const float reflectProbability =
            saturate(dot(fresnel, float3(0.2126, 0.7152, 0.0722)));

        float3 radiance = 0.0;
        float hitDistance = gGi.params6.x;

        if (transmission > 0.001)
        {
            const float pick = Hash01(dispatchThreadId.xy, sampleSeed, 257u);
            if (pick < reflectProbability)
            {
                const float3 direction = reflect(-viewDirection, halfVector);
                radiance = TraceSpecularRay(origin, direction, dispatchThreadId.xy, sampleSeed, hitDistance);
                radiance *= min(fresnel / max(reflectProbability, 1e-4), float3(4.0, 4.0, 4.0));
            }
            else
            {
                // Volume absorption over the material thickness (glTF volume approximation).
                float3 transmittance = 1.0;
                if (attenuationDistance > 0.0 && thickness > 0.0)
                {
                    const float3 sigma = -log(attenuationColor) / attenuationDistance;
                    transmittance = exp(-sigma * thickness * max(gGi.params7.z, 0.0));
                }
                const float refractionWeight = 1.0 / max(1.0 - reflectProbability, 1e-4);

                if (dispersion > 0.0 && gGi.params7.y > 0.5)
                {
                    const float3 incident = -viewDirection;
                    for (int channel = 0; channel < 3; ++channel)
                    {
                        const float channelIor = ior * (1.0 + (float(channel) - 1.0) * dispersion * 0.25);
                        bool totalInternal = false;
                        const float3 direction =
                            RefractAroundMicrofacet(incident, halfVector, channelIor, totalInternal);
                        float channelDistance = gGi.params6.x;
                        const float3 path = TraceSpecularRay(
                            origin,
                            direction,
                            dispatchThreadId.xy,
                            sampleSeed + uint(channel) * 5821u,
                            channelDistance);
                        radiance[channel] =
                            path[channel] * transmittance[channel] * (1.0 - fresnel[channel]) *
                            refractionWeight;
                        if (channel == 0)
                        {
                            hitDistance = channelDistance;
                        }
                    }
                }
                else
                {
                    bool totalInternal = false;
                    const float3 direction =
                        RefractAroundMicrofacet(-viewDirection, halfVector, ior, totalInternal);
                    radiance = TraceSpecularRay(origin, direction, dispatchThreadId.xy, sampleSeed, hitDistance);
                    radiance *= transmittance * (1.0 - fresnel) * refractionWeight;
                }
            }
        }
        else
        {
            const float3 direction = reflect(-viewDirection, halfVector);
            radiance = TraceSpecularRay(origin, direction, dispatchThreadId.xy, sampleSeed, hitDistance);
            radiance *= fresnel;
        }

        radianceAccum += radiance;
        hitDistanceAccum += hitDistance;
    }

    const float inverseSamples = 1.0 / float(sampleCount);
    gReflectionOut[halfPixel] =
        float4(clamp(radianceAccum * inverseSamples, 0.0, kMaxGiRadiance), 1.0);
    gHitDistanceOut[halfPixel] = hitDistanceAccum * inverseSamples;
}
