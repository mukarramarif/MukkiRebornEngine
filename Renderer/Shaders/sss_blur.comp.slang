// Separable, depth-aware diffusion blur for KHR_materials_diffuse_transmission. The pass is
// dispatched once per direction with the source/destination sizes in the push constant, so the
// first horizontal pass downsamples the full-resolution SSS source to half resolution and the
// following passes ping-pong between the two half-resolution targets.

struct SssUniform
{
    float4 params;     // x: radiusPx, y: depth tolerance scale, z: tap spacing px, w: unused
    float4 fullSize;   // xy: full resolution
    float4 sourceSize; // xy: source resolution
    float4 destSize;   // xy: destination resolution
    float4 direction;  // xy: (1, 0) horizontal or (0, 1) vertical
    float4 cameraPosition;
    float4 projection; // x: tan(fovY / 2), y: full resolution height
    float4 lutParams;  // x: 1 / log(range), y: -log(min) / log(range)
};

[[vk::push_constant]] ConstantBuffer<SssUniform> gSss;
[[vk::binding(0, 0)]] Texture2D gSource;
[[vk::binding(1, 0)]] SamplerState gSampler;
[[vk::binding(2, 0)]] Texture2D gPosition;
[[vk::binding(3, 0)]] Texture2D<float4> gLut;
[[vk::binding(4, 0)]] SamplerState gLutSampler;
[[vk::binding(5, 0)]] RWTexture2D<float4> gDest;

static const int kSssTapCount = 8;

[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 pixel = int2(dispatchThreadId.xy);
    const int2 destMax = int2(gSss.destSize.xy) - 1;
    if (pixel.x > destMax.x || pixel.y > destMax.y)
    {
        return;
    }

    const float2 uv = (float2(pixel) + 0.5) / gSss.destSize.xy;
    const float4 center = gSource.SampleLevel(gSampler, uv, 0.0);
    if (center.a < 0.5)
    {
        // Not a subsurface pixel: keep the source value so later passes can still gather it.
        gDest[pixel] = center;
        return;
    }

    const int2 fullMax = int2(gSss.fullSize.xy) - 1;
    const int2 centerFullPixel = clamp(
        int2(uv * gSss.fullSize.xy),
        int2(0, 0),
        fullMax);
    const float3 centerPosition = gPosition.Load(int3(centerFullPixel, 0)).xyz;
    const float viewDistance = max(distance(centerPosition, gSss.cameraPosition.xyz), 1.0e-3);

    // World size of one pixel at the center depth, used to convert the depth tolerance into a
    // world-space comparison. Taps that land on a different surface (large world jump) are faded.
    const float pixelWorld = 2.0 * gSss.projection.x * viewDistance / max(gSss.projection.y, 1.0);
    const float targetRadius = max(gSss.params.x, 1.0);
    const float tolerance = max(gSss.params.y * targetRadius * pixelWorld, 1.0e-4);
    const float spacing = max(gSss.params.z, 0.5);
    const float bucket = saturate(log(targetRadius) * gSss.lutParams.x + gSss.lutParams.y);

    float3 accum = float3(0.0);
    float3 weightSum = float3(0.0);
    [unroll]
    for (int tap = -kSssTapCount; tap <= kSssTapCount; ++tap)
    {
        const float2 tapUv =
            uv + gSss.direction.xy * (float(tap) * spacing) / gSss.destSize.xy;
        if (tapUv.x < 0.0 || tapUv.y < 0.0 || tapUv.x > 1.0 || tapUv.y > 1.0)
        {
            continue;
        }

        const float4 tapSample = gSource.SampleLevel(gSampler, tapUv, 0.0);
        const int2 tapFullPixel = clamp(
            int2(tapUv * gSss.fullSize.xy),
            int2(0, 0),
            fullMax);
        const float3 tapPosition = gPosition.Load(int3(tapFullPixel, 0)).xyz;
        const float spatial = exp(-abs(distance(centerPosition, tapPosition)) / tolerance);

        // The LUT row is selected by the pixel's scattering radius and the column by the normalized
        // tap distance, giving per-channel diffusion weights.
        const float distancePx = abs(float(tap)) * spacing;
        const float normalizedDistance = saturate(distancePx / targetRadius);
        const float3 profile =
            gLut.SampleLevel(gLutSampler, float2(normalizedDistance, bucket), 0.0).rgb;

        // Taps outside the subsurface coverage must not dilute the result (no dark seams).
        const float coverage = step(0.5, tapSample.a);
        const float3 weight = profile * spatial * coverage;
        accum += tapSample.rgb * weight;
        weightSum += weight;
    }

    const float3 result = (weightSum.x + weightSum.y + weightSum.z) > 1.0e-6
                              ? accum / max(weightSum, 1.0e-5)
                              : center.rgb;
    gDest[pixel] = float4(result, center.a);
}
