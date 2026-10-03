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
    float4 params7; // x: glassRefractionBounces, y: dispersionEnabled, z: volumeAbsorptionScale, w: unused
    float4 reflectionSize; // x/y: reflection trace resolution
};

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] Texture2D gDepth;
[[vk::binding(2, 0)]] Texture2D gNormal;
[[vk::binding(3, 0)]] Texture2D gMotion;
[[vk::binding(4, 0)]] Texture2D gReflectionNoisy;
[[vk::binding(5, 0)]] Texture2D gReflectionHistory;
[[vk::binding(6, 0)]] Texture2D gReflectionMoments;
[[vk::binding(7, 0)]] RWTexture2D<float4> gReflectionHistoryOut;
[[vk::binding(8, 0)]] RWTexture2D<float4> gReflectionMomentsOut;

static const float kMaxGiRadiance = 20.0;

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

bool ValidateHistory(float depth, float3 normal, float2 previousUv)
{
    if (any(previousUv < 0.0) || any(previousUv > 1.0))
    {
        return false;
    }

    const int2 previousPixel =
        clamp(int2(previousUv * gGi.screenSize.xy), int2(0, 0), int2(gGi.screenSize.xy) - 1);
    const float previousDepth = gDepth.Load(int3(previousPixel, 0)).r;
    if (previousDepth <= 0.0)
    {
        return false;
    }

    if (depth > previousDepth * 1.1 || previousDepth > depth * 1.1)
    {
        return false;
    }

    const float3 previousNormal = normalize(gNormal.Load(int3(previousPixel, 0)).xyz * 2.0 - 1.0);
    return dot(normal, previousNormal) > 0.9;
}

// Single signal temporal accumulation for the reflection pass, mirroring the GI denoiser.
[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 halfPixel = int2(dispatchThreadId.xy);
    if (halfPixel.x >= int(gGi.reflectionSize.x) || halfPixel.y >= int(gGi.reflectionSize.y))
    {
        return;
    }

    const int2 halfMax = int2(gGi.reflectionSize.xy) - 1;
    const float2 uv = (float2(halfPixel) + 0.5) / gGi.reflectionSize.xy;
    const int2 fullPixel = clamp(int2(uv * gGi.screenSize.xy), int2(0, 0), int2(gGi.screenSize.xy) - 1);

    const float depth = gDepth.Load(int3(fullPixel, 0)).r;
    const float3 normal = normalize(gNormal.Load(int3(fullPixel, 0)).xyz * 2.0 - 1.0);
    const float2 motion = gMotion.Load(int3(fullPixel, 0)).xy;
    const float2 previousUv = (float2(fullPixel) + 0.5 - motion) / gGi.screenSize.xy;

    const float3 currentNoisy = Sanitize(gReflectionNoisy.Load(int3(halfPixel, 0)).rgb);
    const bool resetHistory = gGi.params5.w > 0.5;
    if (resetHistory || depth <= 0.0 || depth >= 1.0 || !ValidateHistory(depth, normal, previousUv))
    {
        const float lum = Luminance(currentNoisy);
        gReflectionHistoryOut[halfPixel] = float4(currentNoisy, 1.0);
        gReflectionMomentsOut[halfPixel] = float4(lum, lum * lum, 1.0, 0.0);
        return;
    }

    const float2 previousHalf = previousUv * gGi.reflectionSize.xy - 0.5;
    const int2 historyPixel = clamp(int2(round(previousHalf)), int2(0, 0), halfMax);
    const float4 historySample = gReflectionHistory.Load(int3(historyPixel, 0));
    const float4 momentsSample = gReflectionMoments.Load(int3(historyPixel, 0));
    const bool historyFinite =
        all(isfinite(historySample.rgb)) && all(isfinite(momentsSample.xyz));

    float3 current = currentNoisy;
    if (historyFinite)
    {
        const float currentLum = Luminance(current);
        const float historyMean = momentsSample.x;
        const float historyStd = sqrt(max(momentsSample.y - historyMean * historyMean, 0.0));
        const float limit = historyMean + 4.0 * max(historyStd, 1e-3);
        if (currentLum > limit)
        {
            current *= limit / max(currentLum, 1e-6);
        }
    }

    float3 minColor = current;
    float3 maxColor = current;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const float3 neighbor =
                Sanitize(gReflectionNoisy.Load(int3(clamp(halfPixel + int2(x, y), int2(0, 0), halfMax), 0)).rgb);
            minColor = min(minColor, neighbor);
            maxColor = max(maxColor, neighbor);
        }
    }

    if (!historyFinite)
    {
        const float resetLum = Luminance(current);
        gReflectionHistoryOut[halfPixel] = float4(current, 1.0);
        gReflectionMomentsOut[halfPixel] = float4(resetLum, resetLum * resetLum, 1.0, 0.0);
        return;
    }

    const float count = clamp(momentsSample.z, 0.0, 64.0);
    const float alpha = clamp(max(1.0 / (count + 1.0), gGi.params2.x), 0.0, 1.0);
    const float3 clamped = clamp(historySample.rgb, minColor, maxColor);
    const float3 color = lerp(clamped, current, alpha);

    const float lum = Luminance(current);
    const float mean = lerp(momentsSample.x, lum, alpha);
    const float mean2 = lerp(momentsSample.y, lum * lum, alpha);

    gReflectionHistoryOut[halfPixel] = float4(Sanitize(color), 1.0);
    gReflectionMomentsOut[halfPixel] = float4(mean, mean2, min(count + 1.0, 64.0), 0.0);
}
