// Temporal reprojection for the cloud signal. Sky rays are reprojected by reconstructing the world
// point at the current cloud distance and projecting it with the previous frame's view-projection.
// Mirrors mukki::CloudUniform in Renderer/Clouds/CloudSettings.h.

struct CloudUniform
{
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    float4x4 viewProjection;
    float4 cameraPosition;
    float4 screenSize;     // x/y: full res, z/w: cloud trace res
    float4 sunDirectionIntensity;
    float4 sunColor;
    float4 cloudParams;
    float4 cloudParams2;
    float4 cloudParams3;
    float4 skySh[9];
    float4 skyParams;
    float4 noiseScales;
    float4 temporalParams; // x: history blend, y: reset flag
    float4 cloudParams4;
};

[[vk::binding(0, 0)]] ConstantBuffer<CloudUniform> gCloud;
[[vk::binding(1, 0)]] Texture2D<float4> gCloudCurrent;
[[vk::binding(2, 0)]] Texture2D<float> gCloudCurrentDistance;
[[vk::binding(3, 0)]] Texture2D<float> gCloudCurrentMask;
[[vk::binding(4, 0)]] Texture2D<float4> gCloudHistory;
[[vk::binding(5, 0)]] Texture2D<float> gCloudHistoryDistance;
[[vk::binding(6, 0)]] Texture2D<float> gCloudHistoryMask;
[[vk::binding(7, 0)]] RWTexture2D<float4> gCloudHistoryOut;
[[vk::binding(8, 0)]] RWTexture2D<float> gCloudHistoryDistanceOut;
[[vk::binding(9, 0)]] RWTexture2D<float> gCloudHistoryMaskOut;

[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 pixel = int2(dispatchThreadId.xy);
    if (pixel.x >= int(gCloud.screenSize.z) || pixel.y >= int(gCloud.screenSize.w))
    {
        return;
    }

    const int2 maxPixel = int2(gCloud.screenSize.zw) - 1;
    const float4 current = gCloudCurrent.Load(int3(pixel, 0));
    const float currentDistance = gCloudCurrentDistance.Load(int3(pixel, 0)).r;
    const float currentMask = gCloudCurrentMask.Load(int3(pixel, 0)).r;

    const bool reset = gCloud.temporalParams.y > 0.5;
    float4 history = current;
    float historyDistance = currentDistance;
    float historyMask = currentMask;
    bool valid = false;

    if (!reset && currentDistance > 1.0)
    {
        // Reconstruct the world point at the current cloud distance and reproject it.
        const float2 uv = (float2(pixel) + 0.5) / gCloud.screenSize.zw;
        const float2 ndc = uv * 2.0 - 1.0;
        const float4 farClip = mul(gCloud.inverseViewProjection, float4(ndc, 1.0, 1.0));
        const float3 farPoint = farClip.xyz / max(farClip.w, 1e-6);
        const float3 rayDirection = normalize(farPoint - gCloud.cameraPosition.xyz);
        const float3 worldPosition = gCloud.cameraPosition.xyz + rayDirection * currentDistance;

        const float4 previousClip = mul(gCloud.previousViewProjection, float4(worldPosition, 1.0));
        if (previousClip.w > 0.0)
        {
            // Projection flips y and the viewport has a positive height: negative NDC y is the top
            // of the framebuffer (Vulkan), so NDC -> uv must not negate y.
            const float2 previousUv =
                float2(previousClip.x / previousClip.w, previousClip.y / previousClip.w) * 0.5 + 0.5;
            if (all(previousUv >= 0.0) && all(previousUv <= 1.0))
            {
                const int2 previousPixel = clamp(
                    int2(previousUv * gCloud.screenSize.zw),
                    int2(0, 0),
                    maxPixel);
                const float previousDistance =
                    gCloudHistoryDistance.Load(int3(previousPixel, 0)).r;
                // Accept the history when it belongs to the same cloud sheet.
                if (previousDistance > 1.0 &&
                    abs(previousDistance - currentDistance) <
                        max(0.2 * currentDistance, 200.0))
                {
                    history = gCloudHistory.Load(int3(previousPixel, 0));
                    historyDistance = previousDistance;
                    historyMask = gCloudHistoryMask.Load(int3(previousPixel, 0)).r;
                    valid = true;
                }
            }
        }
    }

    const float blend = valid ? clamp(gCloud.temporalParams.x, 0.0, 0.98) : 0.0;
    const float4 resolved = lerp(current, history, blend);
    gCloudHistoryOut[pixel] = resolved;
    gCloudHistoryDistanceOut[pixel] = lerp(currentDistance, historyDistance, blend);
    gCloudHistoryMaskOut[pixel] = lerp(currentMask, historyMask, blend);
}
