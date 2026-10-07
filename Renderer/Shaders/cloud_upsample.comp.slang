// Depth/distance aware upsample of the half-resolution cloud signal to full resolution.
// Mirrors mukki::CloudUniform in Renderer/Clouds/CloudSettings.h.

struct CloudUniform
{
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    float4x4 viewProjection;
    float4 cameraPosition;
    float4 screenSize;
    float4 sunDirectionIntensity;
    float4 sunColor;
    float4 cloudParams;
    float4 cloudParams2;
    float4 cloudParams3;
    float4 skySh[9];
    float4 skyParams;
    float4 noiseScales;
    float4 temporalParams;
    float4 cloudParams4;
};

[[vk::binding(0, 0)]] ConstantBuffer<CloudUniform> gCloud;
[[vk::binding(1, 0)]] Texture2D<float4> gCloudIn;
[[vk::binding(2, 0)]] Texture2D<float> gCloudDistance;
[[vk::binding(3, 0)]] Texture2D<float> gDepth;
[[vk::binding(4, 0)]] RWTexture2D<float4> gCloudOut;

[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 fullPixel = int2(dispatchThreadId.xy);
    if (fullPixel.x >= int(gCloud.screenSize.x) || fullPixel.y >= int(gCloud.screenSize.y))
    {
        return;
    }

    const int2 cloudMax = int2(gCloud.screenSize.zw) - 1;
    const int2 fullMax = int2(gCloud.screenSize.xy) - 1;

    const float centerDepth = gDepth.Load(int3(fullPixel, 0)).r;
    const float2 cloudPosition =
        (float2(fullPixel) + 0.5) * (gCloud.screenSize.zw / gCloud.screenSize.xy) - 0.5;
    const int2 cloudBase = int2(floor(cloudPosition));

    // Distance of the nearest cloud tap acts as the edge-stop reference.
    const int2 referencePixel = clamp(cloudBase, int2(0, 0), cloudMax);
    const float referenceDistance = gCloudDistance.Load(int3(referencePixel, 0)).r;

    float4 sum = 0.0;
    float weightSum = 0.0;
    for (int y = 0; y <= 1; ++y)
    {
        for (int x = 0; x <= 1; ++x)
        {
            const int2 cloudPixel = clamp(cloudBase + int2(x, y), int2(0, 0), cloudMax);
            const int2 tapFull = clamp(
                int2(floor((float2(cloudPixel) + 0.5) * (gCloud.screenSize.xy / gCloud.screenSize.zw))),
                int2(0, 0),
                fullMax);
            const float tapDepth = gDepth.Load(int3(tapFull, 0)).r;
            const float tapDistance = gCloudDistance.Load(int3(cloudPixel, 0)).r;

            float weight = 1.0;
            // Sky pixels have equal depth; the cloud distance keeps separate cloud sheets apart.
            weight *= exp(-abs(tapDepth - centerDepth) * 1024.0);
            weight *= exp(-abs(tapDistance - referenceDistance) * 0.01);

            sum += gCloudIn.Load(int3(cloudPixel, 0)) * weight;
            weightSum += weight;
        }
    }

    gCloudOut[fullPixel] = weightSum > 0.0 ? sum / weightSum : gCloudIn.Load(int3(clamp(cloudBase, int2(0, 0), cloudMax), 0));
}
