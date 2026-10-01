struct GiUniform
{
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    float4x4 viewProjection;
    float4 screenSize;
    float4 halfSize;
    float4 cameraPosition;
    float4 skyColor;
    float4 sunDirectionIntensity;
    float4 sunColor;
    float4 params;  // x: maxDistance, y: frameCounter, z: specularMinRoughness, w: feedbackIntensity
    float4 params2; // x: temporalAlphaMin, y: phiDepth, z: phiNormal, w: phiLuminance
    float4 params3; // x: phiHitDistance, y: intensity, z: sunShadowRay, w: resetHistory
    float4 params4; // x: sunShadowSamples, y: sunAngularRadius (radians), z: traceSamples, w: rtSunMask
    float4 params5; // x: probeRays, y: probeHysteresis, z: probeCount, w: unused
};

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] Texture2D gDepth;
[[vk::binding(2, 0)]] Texture2D gNormal;
[[vk::binding(3, 0)]] Texture2D gMotion;
[[vk::binding(4, 0)]] Texture2D gDiffuseNoisy;
[[vk::binding(5, 0)]] Texture2D gSpecularNoisy;
[[vk::binding(6, 0)]] Texture2D gDiffuseHistory;
[[vk::binding(7, 0)]] Texture2D gSpecularHistory;
[[vk::binding(8, 0)]] Texture2D gDiffuseMoments;
[[vk::binding(9, 0)]] Texture2D gSpecularMoments;
[[vk::binding(10, 0)]] RWTexture2D<float4> gDiffuseHistoryOut;
[[vk::binding(11, 0)]] RWTexture2D<float4> gSpecularHistoryOut;
[[vk::binding(12, 0)]] RWTexture2D<float4> gDiffuseMomentsOut;
[[vk::binding(13, 0)]] RWTexture2D<float4> gSpecularMomentsOut;

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

void AccumulateSignal(
    Texture2D noisy,
    Texture2D history,
    Texture2D moments,
    RWTexture2D<float4> historyOut,
    RWTexture2D<float4> momentsOut,
    int2 halfPixel,
    float2 previousUv)
{
    const int2 halfMax = int2(gGi.halfSize.xy) - 1;

    const float2 previousHalf = previousUv * gGi.halfSize.xy - 0.5;
    const int2 historyPixel = clamp(int2(round(previousHalf)), int2(0, 0), halfMax);
    const float4 historySample = history.Load(int3(historyPixel, 0));
    const float4 momentsSample = moments.Load(int3(historyPixel, 0));
    const bool historyFinite =
        all(isfinite(historySample.rgb)) && all(isfinite(momentsSample.xyz));

    float3 current = Sanitize(noisy.Load(int3(halfPixel, 0)).rgb);

    // Reject temporal fireflies: a sample far outside the accumulated mean is pulled back to
    // mean + 4 sigma so it cannot pollute the history or the neighborhood bounds.
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
                Sanitize(noisy.Load(int3(clamp(halfPixel + int2(x, y), int2(0, 0), halfMax), 0)).rgb);
            minColor = min(minColor, neighbor);
            maxColor = max(maxColor, neighbor);
        }
    }

    if (!historyFinite)
    {
        const float resetLum = Luminance(current);
        historyOut[halfPixel] = float4(current, 1.0);
        momentsOut[halfPixel] = float4(resetLum, resetLum * resetLum, 1.0, 0.0);
        return;
    }

    const float count = clamp(momentsSample.z, 0.0, 64.0);
    const float alpha = clamp(max(1.0 / (count + 1.0), gGi.params2.x), 0.0, 1.0);
    const float3 clamped = clamp(historySample.rgb, minColor, maxColor);
    const float3 color = lerp(clamped, current, alpha);

    const float lum = Luminance(current);
    const float mean = lerp(momentsSample.x, lum, alpha);
    const float mean2 = lerp(momentsSample.y, lum * lum, alpha);

    historyOut[halfPixel] = float4(Sanitize(color), 1.0);
    momentsOut[halfPixel] = float4(mean, mean2, min(count + 1.0, 64.0), 0.0);
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
    const float3 normal =
        normalize(gNormal.Load(int3(fullPixel, 0)).xyz * 2.0 - 1.0);
    const float2 motion = gMotion.Load(int3(fullPixel, 0)).xy;
    const float2 previousUv = (float2(fullPixel) + 0.5 - motion) / gGi.screenSize.xy;

    const bool resetHistory = gGi.params3.w > 0.5;
    if (resetHistory || depth <= 0.0 || depth >= 1.0 || !ValidateHistory(depth, normal, previousUv))
    {
        const float3 diffuse = Sanitize(gDiffuseNoisy.Load(int3(halfPixel, 0)).rgb);
        const float3 specular = Sanitize(gSpecularNoisy.Load(int3(halfPixel, 0)).rgb);
        const float diffuseLum = Luminance(diffuse);
        const float specularLum = Luminance(specular);

        gDiffuseHistoryOut[halfPixel] = float4(diffuse, 1.0);
        gSpecularHistoryOut[halfPixel] = float4(specular, 1.0);
        gDiffuseMomentsOut[halfPixel] = float4(diffuseLum, diffuseLum * diffuseLum, 1.0, 0.0);
        gSpecularMomentsOut[halfPixel] = float4(specularLum, specularLum * specularLum, 1.0, 0.0);
        return;
    }

    AccumulateSignal(
        gDiffuseNoisy,
        gDiffuseHistory,
        gDiffuseMoments,
        gDiffuseHistoryOut,
        gDiffuseMomentsOut,
        halfPixel,
        previousUv);
    AccumulateSignal(
        gSpecularNoisy,
        gSpecularHistory,
        gSpecularMoments,
        gSpecularHistoryOut,
        gSpecularMomentsOut,
        halfPixel,
        previousUv);
}
