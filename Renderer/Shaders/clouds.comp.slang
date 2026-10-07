// Nubis-style volumetric cloud raymarch (Decima). Traces a half-resolution (configurable) cloud
// layer, lighting samples with a short sun march and writing scattering/transmittance, a god-ray
// mask and an estimated cloud distance for the temporal reprojection.
//
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
    float4 cloudParams;  // x: coverage, y: cloud type, z: density, w: detail strength
    float4 cloudParams2; // x: bottom km, y: top km, z: altitude variation km, w: curl
    float4 cloudParams3; // x: wind direction radians, y: wind speed, z: time seconds
    float4 skySh[9];
    float4 skyParams;    // x: ambient scale, y: coverage variation
    float4 noiseScales;  // x: base, y: detail, z: curl, w: weather (1 / metres)
    float4 temporalParams; // x: history blend, y: reset flag
    float4 cloudParams4;   // x: primary steps, y: light steps, z: frame, w: unused
};

[[vk::binding(0, 0)]] ConstantBuffer<CloudUniform> gCloud;
[[vk::binding(1, 0)]] Texture2D<float> gDepth;
[[vk::binding(2, 0)]] Texture3D<float4> gCloudBase;
[[vk::binding(3, 0)]] Texture3D<float4> gCloudDetail;
[[vk::binding(4, 0)]] Texture2D<float4> gCloudCurl;
[[vk::binding(5, 0)]] Texture2D<float4> gCloudWeather;
[[vk::binding(6, 0)]] RWTexture2D<float4> gCloudOut;     // rgb scattering, a transmittance
[[vk::binding(7, 0)]] RWTexture2D<float> gCloudMask;     // god ray mask
[[vk::binding(8, 0)]] RWTexture2D<float> gCloudDistance; // estimated cloud distance
[[vk::binding(9, 0)]] SamplerState gCloudSampler;

static const float kPi = 3.14159265359;
static const uint kMaxCloudSteps = 128;

static const float3 kConeOffsets[6] = {
    float3(0.0, 0.0, 0.0),
    float3(0.6, 0.0, 0.0),
    float3(-0.4, 0.4, 0.0),
    float3(0.2, -0.5, 0.0),
    float3(-0.5, -0.2, 0.0),
    float3(0.0, 0.0, 0.0)};

float Remap(float value, float inMin, float inMax, float outMin, float outMax)
{
    return outMin + clamp((value - inMin) / max(inMax - inMin, 1e-5), 0.0, 1.0) * (outMax - outMin);
}

uint HashUint(uint2 pixel, uint frame, uint salt)
{
    uint hash = pixel.x * 1973u + pixel.y * 9277u + frame * 26699u + salt * 7193u;
    hash ^= hash >> 16;
    hash *= 0x7FEB352Du;
    hash ^= hash >> 15;
    hash *= 0x846CA68Bu;
    hash ^= hash >> 16;
    return hash;
}

float Hash01(uint2 pixel, uint frame, uint salt)
{
    return float(HashUint(pixel, frame, salt)) * (1.0 / 4294967296.0);
}

float3 EvaluateSkySh(float3 direction)
{
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    return gCloud.skySh[0].rgb * 0.282095 +
           gCloud.skySh[1].rgb * (0.488603 * x) +
           gCloud.skySh[2].rgb * (0.488603 * y) +
           gCloud.skySh[3].rgb * (0.488603 * z) +
           gCloud.skySh[4].rgb * (1.092548 * x * y) +
           gCloud.skySh[5].rgb * (1.092548 * y * z) +
           gCloud.skySh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) +
           gCloud.skySh[7].rgb * (1.092548 * x * z) +
           gCloud.skySh[8].rgb * (0.546274 * (x * x - y * y));
}

// Cloud type height gradients: (fade in start, full start, full end, fade out end).
float4 TypeGradient(float cloudType)
{
    const float4 stratus = float4(0.0, 0.15, 0.45, 0.55);
    const float4 cumulus = float4(0.05, 0.25, 0.55, 0.8);
    const float4 cumulonimbus = float4(0.0, 0.1, 0.45, 1.0);
    const float4 cirrus = float4(0.1, 0.3, 0.6, 0.85);

    const float t = clamp(cloudType, 0.0, 1.0) * 3.0;
    if (t < 1.0)
    {
        return lerp(stratus, cumulus, t);
    }
    if (t < 2.0)
    {
        return lerp(cumulus, cumulonimbus, t - 1.0);
    }
    return lerp(cumulonimbus, cirrus, t - 2.0);
}

float HeightGradient(float height01, float cloudType)
{
    const float4 gradient = TypeGradient(cloudType);
    const float lower = smoothstep(gradient.x, max(gradient.y, gradient.x + 1e-4), height01);
    const float upper = 1.0 - smoothstep(gradient.z, max(gradient.w, gradient.z + 1e-4), height01);
    return lower * upper;
}

// Wind offset in noise space, shared by the base/detail/curl sampling.
float3 WindOffset(float time)
{
    const float2 direction = float2(cos(gCloud.cloudParams3.x), sin(gCloud.cloudParams3.x));
    return float3(direction.x, 0.0, direction.y) * (gCloud.cloudParams3.y * time);
}

// Cheap base density used by the light march (no detail erosion).
float CheapDensity(float3 position, float3 windOffset)
{
    const float3 baseCoord = (position + windOffset) * gCloud.noiseScales.x;
    const float4 baseSample = gCloudBase.SampleLevel(gCloudSampler, baseCoord, 0.0);

    const float lowFreqFbm = baseSample.g * 0.625 + baseSample.b * 0.25 + baseSample.a * 0.125;
    const float coverage = gCloud.cloudParams.x;
    float baseCloud = Remap(baseSample.r, lowFreqFbm - 1.0 + coverage, 1.0, 0.0, 1.0);
    return baseCloud;
}

// Full density: base shape, height gradient, curl distortion and detail erosion.
float SampleDensity(
    float3 position,
    float height01,
    float3 windOffset,
    out float extinction)
{
    extinction = 0.0;
    if (height01 <= 0.0 || height01 >= 1.0)
    {
        return 0.0;
    }

    const float3 scroll = windOffset * 0.00004;
    float3 baseCoord = (position + windOffset) * gCloud.noiseScales.x + scroll;

    // Curl distortion (sampled at the base scale, with its own scrolling).
    const float3 curl = gCloudCurl
                            .SampleLevel(
                                gCloudSampler,
                                baseCoord.xz * gCloud.noiseScales.z + scroll.xz,
                                0.0)
                            .rgb;
    const float3 curlOffset = (curl * 2.0 - 1.0) * gCloud.cloudParams2.w * 0.15;
    baseCoord += curlOffset;

    const float4 baseSample = gCloudBase.SampleLevel(gCloudSampler, baseCoord, 0.0);
    const float lowFreqFbm = baseSample.g * 0.625 + baseSample.b * 0.25 + baseSample.a * 0.125;
    const float coverage = gCloud.cloudParams.x;
    float baseCloud = Remap(baseSample.r, lowFreqFbm - 1.0 + coverage, 1.0, 0.0, 1.0);
    baseCloud *= HeightGradient(height01, gCloud.cloudParams.y);
    if (baseCloud <= 0.001)
    {
        return 0.0;
    }

    const float3 detailCoord = baseCoord * (gCloud.noiseScales.y / gCloud.noiseScales.x) + curlOffset;
    const float3 detailSample = gCloudDetail.SampleLevel(gCloudSampler, detailCoord, 0.0).rgb;
    const float detailFbm = detailSample.r * 0.625 + detailSample.g * 0.25 + detailSample.b * 0.125;
    // Erode the bottoms more (detail appears at the edges/bottoms of the cloud).
    const float erosion = lerp(1.0 - detailFbm, detailFbm, clamp(height01 * 8.0, 0.0, 1.0));

    const float density = Remap(baseCloud, erosion * gCloud.cloudParams.w, 1.0, 0.0, 1.0);
    extinction = density * gCloud.cloudParams.z;
    return density;
}

float LightMarchOpticalDepth(
    float3 position,
    float top,
    float3 sunDirection,
    float3 windOffset)
{
    const uint lightSteps = clamp(uint(gCloud.cloudParams4.y + 0.5), 1u, 8u);
    const float lightStep = max((top - position.y) / float(lightSteps), 1.0);
    float opticalDepth = 0.0;
    for (uint step = 0; step < lightSteps; ++step)
    {
        const float3 offset = kConeOffsets[step] * lightStep * 0.25;
        const float3 samplePosition = position + sunDirection * ((float(step) + 0.5) * lightStep) + offset;
        const float density = CheapDensity(samplePosition, windOffset);
        opticalDepth += density * gCloud.cloudParams.z * lightStep;
    }
    return opticalDepth;
}

float HenyeyGreenstein(float cosTheta, float eccentricity)
{
    const float g2 = eccentricity * eccentricity;
    return (1.0 - g2) / (4.0 * kPi * pow(max(1.0 + g2 - 2.0 * eccentricity * cosTheta, 1e-3), 1.5));
}

float DualHenyeyGreenstein(float cosTheta)
{
    return lerp(
               HenyeyGreenstein(cosTheta, 0.8),
               HenyeyGreenstein(cosTheta, -0.5),
               0.5) *
           0.5 +
           HenyeyGreenstein(cosTheta, 0.5) * 0.5;
}

// Layer intersection for flat cloud layers; returns false when the ray misses the layer.
bool IntersectLayer(
    float3 origin,
    float3 direction,
    float bottom,
    float top,
    out float tStart,
    out float tEnd)
{
    const float denominator = direction.y;
    if (abs(denominator) < 1e-4)
    {
        // Horizontal ray: only hits the layer when the camera is inside it.
        if (origin.y < bottom || origin.y > top)
        {
            return false;
        }
        tStart = 0.0;
        tEnd = 1e9;
        return true;
    }

    const float t0 = (bottom - origin.y) / denominator;
    const float t1 = (top - origin.y) / denominator;
    tStart = min(t0, t1);
    tEnd = max(t0, t1);
    tStart = max(tStart, 0.0);
    return tEnd > tStart && tEnd > 0.0;
}

[numthreads(8, 8, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const int2 cloudPixel = int2(dispatchThreadId.xy);
    const uint cloudWidth = uint(gCloud.screenSize.z + 0.5);
    const uint cloudHeight = uint(gCloud.screenSize.w + 0.5);
    if (cloudPixel.x >= int(cloudWidth) || cloudPixel.y >= int(cloudHeight))
    {
        return;
    }

    const int2 fullMax = int2(gCloud.screenSize.xy) - 1;
    const float2 uv = (float2(cloudPixel) + 0.5) / gCloud.screenSize.zw;
    const int2 fullPixel = clamp(int2(uv * gCloud.screenSize.xy), int2(0, 0), fullMax);
    const float depth = gDepth.Load(int3(fullPixel, 0)).r;

    const float4 empty = float4(0.0, 0.0, 0.0, 1.0);
    if (depth < 1.0)
    {
        // Sky-only compositing: geometry occludes the cloudless pixels entirely.
        gCloudOut[cloudPixel] = empty;
        gCloudMask[cloudPixel] = 0.0;
        gCloudDistance[cloudPixel] = 0.0;
        return;
    }

    const float3 sunDirection = gCloud.sunDirectionIntensity.xyz;
    const float sunIntensity = gCloud.sunDirectionIntensity.w;

    // Reconstruct the world ray (the sky depth is 1, so use the far plane point).
    const float2 ndc = uv * 2.0 - 1.0;
    const float4 farClip = mul(gCloud.inverseViewProjection, float4(ndc, 1.0, 1.0));
    const float3 farPoint = farClip.xyz / max(farClip.w, 1e-6);
    const float3 rayDirection = normalize(farPoint - gCloud.cameraPosition.xyz);
    const float3 origin = gCloud.cameraPosition.xyz;

    // Weather sampled at the world position where the ray crosses the nominal layer midpoint, so
    // the coverage/type/altitude pattern stays anchored in the world when the camera moves.
    const float2 windDirection = float2(cos(gCloud.cloudParams3.x), sin(gCloud.cloudParams3.x));
    const float time = gCloud.cloudParams3.z;
    const float3 windOffset = float3(windDirection.x, 0.0, windDirection.y) *
                              (gCloud.cloudParams3.y * time);
    const float nominalMid = (gCloud.cloudParams2.x + gCloud.cloudParams2.y) * 0.5 * 1000.0;
    const float midDistance =
        abs(rayDirection.y) > 1e-4 ? max((nominalMid - origin.y) / rayDirection.y, 0.0) : 0.0;
    const float3 weatherPoint = origin + rayDirection * midDistance;
    const float2 weatherUv = (weatherPoint.xz + windOffset.xz) * gCloud.noiseScales.w;
    const float3 weather = gCloudWeather.SampleLevel(gCloudSampler, weatherUv, 0.0).rgb;
    const float coverage = saturate(gCloud.cloudParams.x + (weather.r - 0.5) * gCloud.skyParams.y);
    const float cloudType = saturate(gCloud.cloudParams.y + (weather.g - 0.5) * gCloud.skyParams.y * 0.5);
    const float altitudeOffset = (weather.b - 0.5) * gCloud.cloudParams2.z * 1000.0;

    const float bottom = gCloud.cloudParams2.x * 1000.0 + altitudeOffset;
    const float top = gCloud.cloudParams2.y * 1000.0 + altitudeOffset;
    if (top <= bottom)
    {
        gCloudOut[cloudPixel] = empty;
        gCloudMask[cloudPixel] = 0.0;
        gCloudDistance[cloudPixel] = 0.0;
        return;
    }

    float tStart = 0.0;
    float tEnd = 0.0;
    if (!IntersectLayer(origin, rayDirection, bottom, top, tStart, tEnd))
    {
        gCloudOut[cloudPixel] = empty;
        gCloudMask[cloudPixel] = 0.0;
        gCloudDistance[cloudPixel] = 0.0;
        return;
    }
    tEnd = min(tEnd, 40000.0);

    const uint stepCount = clamp(uint(gCloud.cloudParams4.x + 0.5), 1u, kMaxCloudSteps);
    const float stepSize = (tEnd - tStart) / float(stepCount);
    const float jitter = Hash01(uint2(cloudPixel), uint(gCloud.cloudParams4.z + 0.5), 7u);
    float t = tStart + stepSize * jitter;

    float3 scattering = 0.0;
    float transmittance = 1.0;
    float mask = 0.0;
    float distanceWeight = 0.0;
    float distanceAccum = 0.0;
    float sunVisibilityAccum = 0.0;

    [loop]
    for (uint step = 0; step < stepCount; ++step)
    {
        if (t > tEnd || transmittance < 0.01)
        {
            break;
        }

        const float3 position = origin + rayDirection * t;
        const float height01 = saturate((position.y - bottom) / (top - bottom));

        float extinction = 0.0;
        const float density = SampleDensity(position, height01, windOffset, extinction);

        if (density > 0.001)
        {
            const float opticalDepth = extinction * stepSize;
            const float sampleTransmittance = exp(-opticalDepth);

            const float lightOpticalDepth =
                LightMarchOpticalDepth(position, top, sunDirection, windOffset);
            // Fake multiple scattering by combining a thick and a thin Beer term.
            const float lightTransmittance =
                exp(-lightOpticalDepth) * 0.6 + exp(-lightOpticalDepth * 0.25) * 0.4;

            const float cosTheta = dot(rayDirection, sunDirection);
            const float phase = DualHenyeyGreenstein(cosTheta);
            const float powder = 2.0 * exp(-extinction * 40.0);
            const float3 ambient = EvaluateSkySh(float3(0.0, 1.0, 0.0)) * gCloud.skyParams.x;

            const float3 luminance =
                gCloud.sunColor.rgb * (sunIntensity * phase * lightTransmittance * powder) + ambient;

            scattering += transmittance * luminance * (1.0 - sampleTransmittance);
            transmittance *= sampleTransmittance;

            const float weight = transmittance * (1.0 - sampleTransmittance);
            distanceAccum += t * weight;
            distanceWeight += weight;
            sunVisibilityAccum += lightTransmittance * weight;
        }

        t += stepSize;
    }

    const float opacity = 1.0 - transmittance;
    mask = distanceWeight > 0.0
               ? clamp(opacity * (sunVisibilityAccum / max(distanceWeight, 1e-4)), 0.0, 1.0)
               : 0.0;

    gCloudOut[cloudPixel] = float4(clamp(scattering, 0.0, 20.0), saturate(transmittance));
    gCloudMask[cloudPixel] = mask;
    gCloudDistance[cloudPixel] = distanceWeight > 0.0 ? distanceAccum / max(distanceWeight, 1e-4) : 0.0;
}
