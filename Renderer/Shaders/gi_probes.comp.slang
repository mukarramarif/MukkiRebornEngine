#include "gi_structs.slang"

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] RaytracingAccelerationStructure gTlas;
[[vk::binding(2, 0)]] StructuredBuffer<GpuGeometryData> gGeometry;
[[vk::binding(3, 0)]] Texture2D gDepth;
[[vk::binding(4, 0)]] Texture2D gFeedback;
[[vk::binding(5, 0)]] StructuredBuffer<GpuGiProbe> gProbeIn;
[[vk::binding(6, 0)]] RWStructuredBuffer<GpuGiProbe> gProbeOut;
[[vk::binding(9, 0)]] Texture2D<float> gRtSunShadow;
[[vk::binding(13, 0)]] StructuredBuffer<GpuLight> gGiLights;

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];

#include "gi_shading.slang"

// One thread per probe: traces a uniform sphere of rays, projects the incoming radiance onto SH L1
// and blends the result with the previous coefficients. The coefficients are premultiplied so the
// lighting pass computes irradiance over pi as `sh0 + shX * n.x + shY * n.y + shZ * n.z`.
[numthreads(64, 1, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint probeIndex = dispatchThreadId.x;
    const uint probeCount = uint(gGi.params5.z);
    if (probeIndex >= probeCount)
    {
        return;
    }

    const GpuGiProbe probe = gProbeIn[probeIndex];
    const float3 origin = probe.position.xyz;
    const uint rayCount = uint(clamp(gGi.params5.x, 1.0, 256.0));
    const float hysteresis = clamp(gGi.params5.y, 0.0, 0.98);
    const uint frame = uint(gGi.params.y);

    float3 sumRadiance = 0.0;
    float3 sumDirectional = 0.0;

    for (uint rayIndex = 0; rayIndex < rayCount; ++rayIndex)
    {
        const uint2 hashPixel = uint2(probeIndex, rayIndex);
        const float u1 = Hash01(hashPixel, frame, 17u);
        const float u2 = Hash01(hashPixel, frame, 29u);
        const float z = 1.0 - 2.0 * u1;
        const float r = sqrt(max(1.0 - z * z, 0.0));
        const float phi = 2.0 * kPi * u2;
        const float3 direction = float3(r * cos(phi), r * sin(phi), z);

        float3 radiance = gGi.skyColor.rgb * gGi.skyColor.w;
        RayQuery<RAY_FLAG_NONE> query;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = 0.0;
        ray.TMax = gGi.params.x;
        query.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
        query.Proceed();
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            radiance = ShadeHit(
                gFeedback,
                query.CommittedInstanceID(),
                query.CommittedPrimitiveIndex(),
                query.CommittedTriangleBarycentrics(),
                origin,
                direction,
                query.CommittedRayT(),
                hashPixel,
                frame);
        }

        sumRadiance += radiance;
        sumDirectional += radiance * direction;
    }

    const float inverseRays = 1.0 / float(rayCount);
    const float3 sh0 = sumRadiance * inverseRays;
    const float3 sh1 = sumDirectional * (2.0 * inverseRays);

    const float blend = 1.0 - hysteresis;
    GpuGiProbe result = probe;
    result.sh0 = float4(lerp(probe.sh0.rgb, sh0, blend), 1.0);
    result.shX = float4(lerp(probe.shX.rgb, sh1.x, blend), 0.0);
    result.shY = float4(lerp(probe.shY.rgb, sh1.y, blend), 0.0);
    result.shZ = float4(lerp(probe.shZ.rgb, sh1.z, blend), 0.0);
    gProbeOut[probeIndex] = result;
}
