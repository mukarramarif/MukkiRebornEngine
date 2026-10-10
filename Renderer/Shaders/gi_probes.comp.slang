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
//
// The trace also records depth statistics: the mean/variance of the hit distances drives the
// Chebyshev visibility test that stops probes from bleeding light through walls, and the fraction
// of committed hits that are backfaces marks probes embedded in geometry (a probe inside a solid
// sees the shell from the inside, so nearly every hit is a backface).
static const float kProbeStuckBackfaceFraction = 0.6;

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
    const float rayLength = max(gGi.params.x, 1e-3);

    float3 sumRadiance = 0.0;
    float3 sumDirectional = 0.0;
    float hitDistanceSum = 0.0;
    float hitDistanceSquaredSum = 0.0;
    uint committedHitCount = 0;
    uint backfaceHitCount = 0;

    for (uint rayIndex = 0; rayIndex < rayCount; ++rayIndex)
    {
        const uint2 hashPixel = uint2(probeIndex, rayIndex);
        const float u1 = Hash01(hashPixel, frame, 17u);
        const float u2 = Hash01(hashPixel, frame, 29u);
        const float z = 1.0 - 2.0 * u1;
        const float r = sqrt(max(1.0 - z * z, 0.0));
        const float phi = 2.0 * kPi * u2;
        const float3 direction = float3(r * cos(phi), r * sin(phi), z);

        float3 radiance = GiSkyRadiance(direction);
        float hitDistance = rayLength;
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
            hitDistance = query.CommittedRayT();
            ++committedHitCount;
            // The hardware facing test accounts for the instance transform (node rotations,
            // mirrored scales), unlike the object-space vertex normals. A probe inside a solid
            // exits through backfaces in nearly every direction.
            if (!query.CommittedTriangleFrontFace())
            {
                ++backfaceHitCount;
            }

            radiance = ShadeHit(
                gFeedback,
                query.CommittedInstanceID(),
                query.CommittedPrimitiveIndex(),
                query.CommittedTriangleBarycentrics(),
                origin,
                direction,
                hitDistance,
                hashPixel,
                frame);
        }

        hitDistanceSum += hitDistance;
        hitDistanceSquaredSum += hitDistance * hitDistance;

        // A single non-finite ray must not poison the probe SH. Sources are floating point edge
        // cases in the shared hit shading (degenerate normals, retroreflection half vectors).
        if (all(isfinite(radiance)))
        {
            sumRadiance += radiance;
            sumDirectional += radiance * direction;
        }
    }

    const float inverseRays = 1.0 / float(rayCount);
    const float3 sh0 = sumRadiance * inverseRays;
    const float3 sh1 = sumDirectional * (2.0 * inverseRays);
    if (!all(isfinite(sh0)) || !all(isfinite(sh1)))
    {
        gProbeOut[probeIndex] = probe;
        return;
    }

    const float blend = 1.0 - hysteresis;
    GpuGiProbe result = probe;
    result.sh0 = float4(lerp(probe.sh0.rgb, sh0, blend), 1.0);
    result.shX = float4(lerp(probe.shX.rgb, sh1.x, blend), 0.0);
    result.shY = float4(lerp(probe.shY.rgb, sh1.y, blend), 0.0);
    result.shZ = float4(lerp(probe.shZ.rgb, sh1.z, blend), 0.0);

    // Depth statistics: snap on the first trace so the visibility test is correct immediately,
    // then smooth like the SH. The validity flag uses the smoothed backface fraction so per-frame
    // ray jitter cannot flap it around the threshold.
    const float meanHitDistance = hitDistanceSum * inverseRays;
    const float meanHitDistanceSquared = hitDistanceSquaredSum * inverseRays;
    const float backfaceFraction =
        committedHitCount > 0 ? float(backfaceHitCount) / float(committedHitCount) : 0.0;
    const float depthBlend = probe.depth.z >= 0.5 ? blend : 1.0;
    const float smoothedBackfaceFraction = lerp(probe.depth.w, backfaceFraction, depthBlend);
    result.depth = float4(
        lerp(probe.depth.x, meanHitDistance, depthBlend),
        lerp(probe.depth.y, meanHitDistanceSquared, depthBlend),
        smoothedBackfaceFraction < kProbeStuckBackfaceFraction ? 1.0 : 0.0,
        smoothedBackfaceFraction);

    gProbeOut[probeIndex] = result;
}
