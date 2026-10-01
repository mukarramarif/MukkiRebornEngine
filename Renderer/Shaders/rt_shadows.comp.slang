struct RtShadowUniform
{
    float4 screenSize;            // x: width, y: height
    float4 sunDirectionIntensity; // xyz: direction toward the sun, w: intensity
    float4 params; // x: samples, y: angular radius (radians), z: frame counter, w: normal bias
};

[[vk::binding(0, 0)]] ConstantBuffer<RtShadowUniform> gRt;
[[vk::binding(1, 0)]] RaytracingAccelerationStructure gTlas;
[[vk::binding(2, 0)]] Texture2D gPosition;
[[vk::binding(3, 0)]] Texture2D gNormal;
[[vk::binding(4, 0)]] RWTexture2D<float> gVisibilityOut;

static const float kPi = 3.14159265359;

uint HashUint(uint2 pixel, uint frame)
{
    uint h = pixel.x * 1973u + pixel.y * 9277u + frame * 26699u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

float Hash01(uint2 pixel, uint frame, uint salt)
{
    return float(HashUint(pixel + uint2(salt * 17u, salt * 131u), frame)) * (1.0 / 4294967296.0);
}

// Traces a small cone of rays toward the sun and writes the unoccluded fraction. The result is
// consumed by the lighting pass instead of the cascaded shadow map when RT sun shadows are on.
[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 pixel = int2(dispatchThreadId.xy);
    if (pixel.x >= int(gRt.screenSize.x) || pixel.y >= int(gRt.screenSize.y))
    {
        return;
    }

    const float4 position = gPosition.Load(int3(pixel, 0));
    if (position.w <= 0.0)
    {
        gVisibilityOut[pixel] = 1.0;
        return;
    }

    const float3 normal = normalize(gNormal.Load(int3(pixel, 0)).xyz * 2.0 - 1.0);
    const float3 sunDirection = gRt.sunDirectionIntensity.xyz;
    const float nDotL = max(dot(normal, sunDirection), 0.0);
    if (nDotL <= 0.0 || gRt.sunDirectionIntensity.w <= 0.0)
    {
        gVisibilityOut[pixel] = 1.0;
        return;
    }

    const uint sampleCount = uint(clamp(gRt.params.x, 1.0, 16.0));
    const float angularRadius = max(gRt.params.y, 0.0);
    // Static per-pixel seeds: the mask has no temporal history, so varying the jitter per frame
    // would make the shadows boil. The lighting pass averages a 3x3 neighborhood on top of this.
    const uint frame = 0u;
    const float3 origin = position.xyz + normal * gRt.params.w;

    const float3 coneTangent = normalize(
        cross(abs(sunDirection.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0),
              sunDirection));
    const float3 coneBitangent = cross(sunDirection, coneTangent);

    float visibility = 0.0;
    for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        float3 direction = sunDirection;
        if (angularRadius > 0.0)
        {
            const float radius = angularRadius * sqrt(Hash01(pixel, frame, 101u + sampleIndex));
            const float phi = 2.0 * kPi * Hash01(pixel, frame, 137u + sampleIndex);
            direction = normalize(
                sunDirection + (coneTangent * cos(phi) + coneBitangent * sin(phi)) * radius);
        }

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
        RayDesc shadowRay;
        shadowRay.Origin = origin;
        shadowRay.Direction = direction;
        shadowRay.TMin = 0.0;
        shadowRay.TMax = 10000.0;
        shadowQuery.TraceRayInline(gTlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
        shadowQuery.Proceed();
        if (shadowQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            visibility += 1.0;
        }
    }

    gVisibilityOut[pixel] = visibility / float(sampleCount);
}
