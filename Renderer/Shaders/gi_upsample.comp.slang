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
    float4 params5; // x: probeRays, y: probeHysteresis, z: probeCount, w: reflectionResetHistory
    float4 params6; // x: reflectionMaxDistance, y: reflectionMaxRoughness, z: reflectionSamples, w: reflectionIntensity
};

[[vk::binding(0, 0)]] ConstantBuffer<GiUniform> gGi;
[[vk::binding(1, 0)]] Texture2D gDiffuseFiltered;
[[vk::binding(2, 0)]] Texture2D gSpecularFiltered;
[[vk::binding(3, 0)]] Texture2D gAlbedo;
[[vk::binding(4, 0)]] Texture2D gNormal;
[[vk::binding(5, 0)]] Texture2D gDepth;
[[vk::binding(6, 0)]] RWTexture2D<float4> gIndirectOut;

static const float kMaxGiRadiance = 20.0;

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
        gIndirectOut[fullPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float4 centerNormalRough = gNormal.Load(int3(fullPixel, 0));
    const float3 centerNormal = normalize(centerNormalRough.xyz * 2.0 - 1.0);
    const float3 albedo = gAlbedo.Load(int3(fullPixel, 0)).rgb;

    const float2 halfPosition = (float2(fullPixel) + 0.5) * (gGi.halfSize.xy / gGi.screenSize.xy) - 0.5;
    const int2 halfBase = int2(floor(halfPosition));
    const int2 halfMax = int2(gGi.halfSize.xy) - 1;
    const int2 fullMax = int2(gGi.screenSize.xy) - 1;

    float3 diffuseSum = 0.0;
    float3 specularSum = 0.0;
    float weightSum = 0.0;

    [unroll]
    for (int y = 0; y <= 1; ++y)
    {
        [unroll]
        for (int x = 0; x <= 1; ++x)
        {
            const int2 halfPixel = clamp(halfBase + int2(x, y), int2(0, 0), halfMax);
            const int2 tapFull = clamp(halfPixel * 2, int2(0, 0), fullMax);
            const float tapDepth = gDepth.Load(int3(tapFull, 0)).r;
            if (tapDepth <= 0.0 || tapDepth >= 1.0)
            {
                continue;
            }

            const float3 tapNormal =
                normalize(gNormal.Load(int3(tapFull, 0)).xyz * 2.0 - 1.0);
            float weight = exp(-abs(centerDepth - tapDepth) / (gGi.params2.y * max(abs(centerDepth), 1e-3) + 1e-6));
            weight *= pow(saturate(dot(centerNormal, tapNormal)), gGi.params2.z);

            const float3 diffuseSample = gDiffuseFiltered.Load(int3(halfPixel, 0)).rgb;
            const float3 specularSample = gSpecularFiltered.Load(int3(halfPixel, 0)).rgb;
            if (!all(isfinite(diffuseSample)) || !all(isfinite(specularSample)))
            {
                continue;
            }

            diffuseSum += clamp(diffuseSample, 0.0, kMaxGiRadiance) * weight;
            specularSum += clamp(specularSample, 0.0, kMaxGiRadiance) * weight;
            weightSum += weight;
        }
    }

    if (weightSum <= 0.0)
    {
        gIndirectOut[fullPixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 indirect = (albedo * diffuseSum + specularSum) / weightSum;
    gIndirectOut[fullPixel] = float4(clamp(indirect * gGi.params3.y, 0.0, kMaxGiRadiance), 1.0);
}
