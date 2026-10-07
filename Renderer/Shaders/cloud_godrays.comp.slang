// Screen-space god rays: radial blur of the cloud sun-occlusion mask away from the sun's screen
// position, tinted by the sun color. Mirrors mukki::CloudUniform in
// Renderer/Clouds/CloudSettings.h.

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
[[vk::binding(1, 0)]] Texture2D<float> gCloudMask;
[[vk::binding(2, 0)]] Texture2D<float> gCloudMaskHistory;
[[vk::binding(3, 0)]] RWTexture2D<float4> gGodRaysOut;

static const uint kGodRayTaps = 48;

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
    const float2 pixelUv = (float2(pixel) + 0.5) / gCloud.screenSize.zw;

    // Sun screen position (y follows the fullscreen uv convention).
    const float3 sunDirection = gCloud.sunDirectionIntensity.xyz;
    const float4 sunClip = mul(
        gCloud.viewProjection,
        float4(gCloud.cameraPosition.xyz + sunDirection * 10000.0, 1.0));
    if (sunClip.w <= 0.0)
    {
        gGodRaysOut[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Projection flips y and the viewport has a positive height: negative NDC y is the top of the
    // framebuffer (Vulkan), so NDC -> uv must not negate y.
    const float2 sunUv = float2(sunClip.x / sunClip.w, sunClip.y / sunClip.w) * 0.5 + 0.5;
    // Fade the effect out as the sun leaves the screen.
    const float edgeDistance = min(
        min(sunUv.x, 1.0 - sunUv.x),
        min(sunUv.y, 1.0 - sunUv.y));
    const float onScreen = 1.0 - smoothstep(-0.15, 0.05, edgeDistance);

    const float2 ray = sunUv - pixelUv;
    const float rayLength = length(ray);
    float accumulation = 0.0;
    float weightSum = 0.0;
    [loop]
    for (uint tap = 0; tap < kGodRayTaps; ++tap)
    {
        const float t = (float(tap) + 0.5) / float(kGodRayTaps);
        // Concentrate samples near the pixel for a soft, decaying streak.
        const float sampleT = t * t;
        const float2 sampleUv = pixelUv + ray * sampleT;
        if (any(sampleUv < 0.0) || any(sampleUv > 1.0))
        {
            continue;
        }

        const float2 clampedUv = clamp(sampleUv, float2(0.0, 0.0), float2(1.0, 1.0));
        const int2 samplePixel = clamp(
            int2(clampedUv * gCloud.screenSize.zw),
            int2(0, 0),
            maxPixel);
        const float mask = lerp(
            gCloudMask.Load(int3(samplePixel, 0)).r,
            gCloudMaskHistory.Load(int3(samplePixel, 0)).r,
            0.5);

        const float weight = 1.0 - t;
        accumulation += mask * weight;
        weightSum += weight;
    }

    const float intensity =
        weightSum > 0.0 ? accumulation / weightSum : 0.0;
    const float3 tinted =
        gCloud.sunColor.rgb * (intensity * onScreen * gCloud.skyParams.z);
    gGodRaysOut[pixel] = float4(tinted, 1.0);
}
