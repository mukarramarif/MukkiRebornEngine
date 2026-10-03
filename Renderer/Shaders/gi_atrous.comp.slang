#include "gi_structs.slang"

struct AtrousPushConstants
{
    uint signal;    // 0: diffuse, 1: specular
    uint iteration;
};

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] Texture2D gDepth;
[[vk::binding(2, 0)]] Texture2D gNormal;
[[vk::binding(3, 0)]] Texture2D gHitDistance;
[[vk::binding(4, 0)]] Texture2D gColorIn;
[[vk::binding(5, 0)]] Texture2D gMoments;
[[vk::binding(6, 0)]] RWTexture2D<float4> gColorOut;
[[vk::push_constant]] ConstantBuffer<AtrousPushConstants> gPush;


float Luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float3 Sanitize(float3 value)
{
    if (!all(isfinite(value)))
    {
        return 0.0;
    }
    return clamp(value, 0.0, kMaxGiRadiance);
}

[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 halfPixel = int2(dispatchThreadId.xy);
    // Signal 2 is the reflection signal: it can run above the GI half resolution, so the tap
    // mapping to full resolution follows the reflection trace size instead of a fixed 2x.
    const bool reflectionPass = gPush.signal >= 2;
    const float2 traceSize = reflectionPass ? gGi.reflectionSize.xy : gGi.halfSize.xy;
    if (halfPixel.x >= int(traceSize.x) || halfPixel.y >= int(traceSize.y))
    {
        return;
    }

    const int2 halfMax = int2(traceSize) - 1;
    const int2 fullMax = int2(gGi.screenSize.xy) - 1;
    const int step = 1 << int(gPush.iteration);
    const float2 fullScale = gGi.screenSize.xy / traceSize;

    const int2 centerFull = reflectionPass
                                ? clamp(int2(floor((float2(halfPixel) + 0.5) * fullScale)), int2(0, 0), fullMax)
                                : clamp(halfPixel * 2, int2(0, 0), fullMax);
    const float centerDepth = gDepth.Load(int3(centerFull, 0)).r;
    if (centerDepth <= 0.0 || centerDepth >= 1.0)
    {
        gColorOut[halfPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float4 centerNormalRough = gNormal.Load(int3(centerFull, 0));
    const float3 centerNormal = normalize(centerNormalRough.xyz * 2.0 - 1.0);
    const float centerRoughness = centerNormalRough.w;
    const float3 centerColor = Sanitize(gColorIn.Load(int3(halfPixel, 0)).rgb);
    const float centerLum = Luminance(centerColor);
    const float4 centerMoments = gMoments.Load(int3(halfPixel, 0));
    const float variance = all(isfinite(centerMoments.xyz))
                               ? max(centerMoments.y - centerMoments.x * centerMoments.x, 1e-4)
                               : 1e-4;
    const float centerHitDistance = gHitDistance.Load(int3(halfPixel, 0)).r;

    float3 sum = 0.0;
    float weightSum = 0.0;

    [unroll]
    for (int y = -2; y <= 2; ++y)
    {
        [unroll]
        for (int x = -2; x <= 2; ++x)
        {
            const int2 tapHalf = clamp(halfPixel + int2(x, y) * step, int2(0, 0), halfMax);
            const int2 tapFull = reflectionPass
                                     ? clamp(int2(floor((float2(tapHalf) + 0.5) * fullScale)), int2(0, 0), fullMax)
                                     : clamp(tapHalf * 2, int2(0, 0), fullMax);

            const float tapDepth = gDepth.Load(int3(tapFull, 0)).r;
            if (tapDepth <= 0.0 || tapDepth >= 1.0)
            {
                continue;
            }

            const float4 tapNormalRough = gNormal.Load(int3(tapFull, 0));
            const float3 tapNormal = normalize(tapNormalRough.xyz * 2.0 - 1.0);
            const float3 rawTapColor = gColorIn.Load(int3(tapHalf, 0)).rgb;
            if (!all(isfinite(rawTapColor)))
            {
                continue;
            }
            // Clamp outliers to the center's variance so fireflies cannot spread through the kernel.
            const float sigma = 3.0 * sqrt(variance);
            const float3 tapColor = clamp(Sanitize(rawTapColor), centerColor - sigma, centerColor + sigma);
            const float tapLum = Luminance(tapColor);

            float weight = exp(-abs(centerDepth - tapDepth) / (gGi.params2.y * max(abs(centerDepth), 1e-3) + 1e-6));
            weight *= pow(saturate(dot(centerNormal, tapNormal)), gGi.params2.z);
            weight *= exp(-abs(centerLum - tapLum) / (gGi.params2.w * sqrt(variance) + 1e-4));

            if (gPush.signal >= 1)
            {
                // Glossy signals additionally reject taps with different roughness or hit distance.
                weight *= exp(-abs(centerRoughness - tapNormalRough.w) * 32.0);
                const float tapHitDistance = gHitDistance.Load(int3(tapHalf, 0)).r;
                const float hitScale = gGi.params3.x * max(centerHitDistance, 0.1) + 1e-4;
                weight *= exp(-abs(centerHitDistance - tapHitDistance) / hitScale);
            }

            sum += tapColor * weight;
            weightSum += weight;
        }
    }

    const float3 result = weightSum > 0.0 ? Sanitize(sum / weightSum) : centerColor;
    gColorOut[halfPixel] = float4(result, 1.0);
}
