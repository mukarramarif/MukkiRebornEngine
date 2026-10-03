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
    float4 params;
    float4 params2;
    float4 params3;
    float4 params4;
    float4 params5;
    float4 params6;
    float4 params7;
    float4 reflectionSize; // x/y: reflection trace resolution
};

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] Texture2D gReflectionFiltered;
[[vk::binding(2, 0)]] Texture2D gDepth;
[[vk::binding(3, 0)]] Texture2D gNormal;
[[vk::binding(4, 0)]] RWTexture2D<float4> gReflectionOut;

// Joint bilateral upsample of the filtered reflection signal. Roughness is part of the weight so
// glossy and smooth surfaces do not bleed into each other.
[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 fullPixel = int2(dispatchThreadId.xy);
    if (fullPixel.x >= int(gGi.screenSize.x) || fullPixel.y >= int(gGi.screenSize.y))
    {
        return;
    }

    const float centerDepth = gDepth.Load(int3(fullPixel, 0)).r;
    if (centerDepth <= 0.0 || centerDepth >= 1.0)
    {
        gReflectionOut[fullPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float4 centerNormalRough = gNormal.Load(int3(fullPixel, 0));
    const float3 centerNormal = normalize(centerNormalRough.xyz * 2.0 - 1.0);

    const float2 halfPosition =
        (float2(fullPixel) + 0.5) * (gGi.reflectionSize.xy / gGi.screenSize.xy) - 0.5;
    const int2 halfBase = int2(floor(halfPosition));
    const int2 halfMax = int2(gGi.reflectionSize.xy) - 1;
    const int2 fullMax = int2(gGi.screenSize.xy) - 1;
    const float2 fullScale = gGi.screenSize.xy / gGi.reflectionSize.xy;

    float3 sum = 0.0;
    float weightSum = 0.0;

    [unroll]
    for (int y = 0; y <= 1; ++y)
    {
        [unroll]
        for (int x = 0; x <= 1; ++x)
        {
            const int2 halfPixel = clamp(halfBase + int2(x, y), int2(0, 0), halfMax);
            const int2 tapFull =
                clamp(int2(floor((float2(halfPixel) + 0.5) * fullScale)), int2(0, 0), fullMax);
            const float tapDepth = gDepth.Load(int3(tapFull, 0)).r;
            if (tapDepth <= 0.0 || tapDepth >= 1.0)
            {
                continue;
            }

            const float4 tapNormalRough = gNormal.Load(int3(tapFull, 0));
            const float3 tapNormal = normalize(tapNormalRough.xyz * 2.0 - 1.0);
            float weight = exp(-abs(centerDepth - tapDepth) / (gGi.params2.y * max(abs(centerDepth), 1e-3) + 1e-6));
            weight *= pow(saturate(dot(centerNormal, tapNormal)), gGi.params2.z);
            weight *= exp(-abs(centerNormalRough.w - tapNormalRough.w) * 32.0);

            const float3 sample = gReflectionFiltered.Load(int3(halfPixel, 0)).rgb;
            if (!all(isfinite(sample)))
            {
                continue;
            }
            sum += clamp(sample, 0.0, 20.0) * weight;
            weightSum += weight;
        }
    }

    if (weightSum <= 0.0)
    {
        gReflectionOut[fullPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    gReflectionOut[fullPixel] = float4(sum / weightSum, 1.0);
}
