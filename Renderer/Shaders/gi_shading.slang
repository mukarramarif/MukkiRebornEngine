// Shared hit shading used by the per-pixel GI trace and the probe update pass. The includer must
// declare GiUniform, GpuGeometryData, GpuLight and the bindings for gGi, gTlas, gGeometry, gDepth,
// gRtSunShadow, gGiLights, gTextures, gSamplers and gBuffers before including this file.

#include "brdf.slang"
#include "sheen_brdf.slang"
#include "sky.slang"

// Environment-aware sky radiance for rays that miss the scene (loaded equirect map or gradient).
// Falls back to the flat sky color when image based lighting is disabled.
float3 GiSkyRadiance(float3 direction)
{
    if (gGi.skyParams.x > 0.5)
    {
        return EnvironmentRadiance(
            direction,
            gGi.skyZenith.rgb,
            gGi.skyHorizon.rgb,
            gGi.skyGround.rgb,
            gGi.skyParams.y,
            uint(gGi.skyMap.x + 0.5),
            uint(gGi.skyMap.y + 0.5),
            gGi.skyMap.z > 0.5);
    }
    return gGi.skyColor.rgb * gGi.skyColor.w;
}

// Diffuse irradiance / pi from the environment SH (or the flat ambient fallback).
float3 GiIrradianceOverPi(float3 normal)
{
    if (gGi.skyParams.x > 0.5)
    {
        return EvaluateSkySh(gGi.skySh, normal);
    }
    return gGi.skyColor.rgb * gGi.skyColor.w;
}

float LoadFloat(ByteAddressBuffer buffer, uint byteOffset)
{
    return asfloat(buffer.Load(byteOffset));
}

float2 LoadFloat2(ByteAddressBuffer buffer, uint byteOffset)
{
    return float2(asfloat(buffer.Load(byteOffset)), asfloat(buffer.Load(byteOffset + 4)));
}

float3 LoadFloat3(ByteAddressBuffer buffer, uint byteOffset)
{
    return float3(
        asfloat(buffer.Load(byteOffset)),
        asfloat(buffer.Load(byteOffset + 4)),
        asfloat(buffer.Load(byteOffset + 8)));
}

uint HashUint(uint2 pixel, uint frame)
{
    uint h = pixel.x * 1973u + pixel.y * 9277u + frame * 26699u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

float Hash01(uint2 pixel, uint frame, uint salt)
{
    return float(HashUint(pixel + uint2(salt * 17u, salt * 131u), frame)) * (1.0 / 4294967296.0);
}

float3 SampleBilinear(Texture2D texture, float2 uv)
{
    const float2 size = gGi.screenSize.xy;
    const float2 pixel = uv * size - 0.5;
    const int2 base = int2(floor(pixel));
    const float2 fraction = pixel - float2(base);
    const int2 maxPixel = int2(size) - 1;

    const float4 c00 = texture.Load(int3(clamp(base, int2(0, 0), maxPixel), 0));
    const float4 c10 = texture.Load(int3(clamp(base + int2(1, 0), int2(0, 0), maxPixel), 0));
    const float4 c01 = texture.Load(int3(clamp(base + int2(0, 1), int2(0, 0), maxPixel), 0));
    const float4 c11 = texture.Load(int3(clamp(base + int2(1, 1), int2(0, 0), maxPixel), 0));

    const float4 top = lerp(c00, c10, fraction.x);
    const float4 bottom = lerp(c01, c11, fraction.x);
    return lerp(top, bottom, fraction.y).rgb;
}

float3 SampleCosineDirection(float3 normal, uint2 pixel, uint frame)
{
    const float3 up = abs(normal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(up, normal));
    const float3 bitangent = cross(normal, tangent);

    const float r1 = Hash01(pixel, frame, 11u);
    const float r2 = Hash01(pixel, frame, 13u);
    const float phi = 2.0 * kPi * r1;
    const float cosTheta = sqrt(max(1.0 - r2, 0.0));
    const float sinTheta = sqrt(r2);

    return normalize(
        tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + normal * cosTheta);
}

float3 SampleGgxHalfVector(float3 viewDirection, float roughness, uint2 pixel, uint frame)
{
    const float3 view = normalize(viewDirection);
    const float3 tangent =
        normalize(cross(abs(view.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), view));
    const float3 bitangent = cross(view, tangent);

    const float a = max(roughness * roughness, 1e-4);
    const float r1 = Hash01(pixel, frame, 3u);
    const float r2 = Hash01(pixel, frame, 7u);
    const float phi = 2.0 * kPi * r1;
    const float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    const float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    return normalize(
        tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + view * cosTheta);
}

float3 SampleGgxDirection(float3 normal, float3 viewDirection, float roughness, uint2 pixel, uint frame)
{
    const float3 halfVector = SampleGgxHalfVector(viewDirection, roughness, pixel, frame);
    return reflect(-normalize(viewDirection), halfVector);
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals" (isotropic). Returns the sampled
// microfacet normal in world space; `alpha` is the squared perceptual roughness. Unlike NDF
// sampling this accounts for the view direction, so grazing-angle lobes stay correct.
float3 SampleGgxVndf(float3 normal, float3 viewDirection, float2 u, float alpha)
{
    const float3 up = abs(normal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(up, normal));
    const float3 bitangent = cross(normal, tangent);
    const float3 view = float3(
        dot(viewDirection, tangent),
        dot(viewDirection, bitangent),
        dot(viewDirection, normal));

    const float3 stretched = normalize(float3(alpha * view.x, alpha * view.y, view.z));
    const float lengthSquared = stretched.x * stretched.x + stretched.y * stretched.y;
    const float3 axis1 = lengthSquared > 0.0
                             ? float3(-stretched.y, stretched.x, 0.0) * rsqrt(lengthSquared)
                             : float3(1.0, 0.0, 0.0);
    const float3 axis2 = cross(stretched, axis1);

    const float radius = sqrt(u.x);
    const float phi = 2.0 * kPi * u.y;
    const float p1 = radius * cos(phi);
    float p2 = radius * sin(phi);
    const float s = 0.5 * (1.0 + stretched.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;

    const float3 localHalf = p1 * axis1 + p2 * axis2 +
                             sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * stretched;
    const float3 microfacet = float3(alpha * localHalf.x, alpha * localHalf.y, max(0.0, localHalf.z));
    return normalize(tangent * microfacet.x + bitangent * microfacet.y + normal * microfacet.z);
}

// Height-correlated Smith masking term for one direction (nDotX = saturate(dot(n, x))).
float SmithGgxG1(float nDotX, float alpha)
{
    const float alphaSquared = alpha * alpha;
    const float cosSquared = nDotX * nDotX;
    return (2.0 * nDotX) /
           max(nDotX + sqrt(alphaSquared + (1.0 - alphaSquared) * cosSquared), 1e-6);
}

// Height-correlated Smith joint masking. For a VNDF-sampled reflection the estimator weight is
// F * G2 / G1 (the cosine and PDF terms cancel), which keeps grazing angles energy correct.
float SmithGgxG2(float nDotV, float nDotL, float alpha)
{
    const float alphaSquared = alpha * alpha;
    const float lambdaV = nDotL * sqrt(nDotV * nDotV * (1.0 - alphaSquared) + alphaSquared);
    const float lambdaL = nDotV * sqrt(nDotL * nDotL * (1.0 - alphaSquared) + alphaSquared);
    return (2.0 * nDotV * nDotL) / max(lambdaV + lambdaL, 1e-6);
}

// VNDF reflection estimator weight (Fresnel already applied by the caller when needed).
float SpecularVndfWeight(float nDotV, float nDotL, float alpha)
{
    return SmithGgxG2(nDotV, nDotL, alpha) / max(SmithGgxG1(nDotV, alpha), 1e-4);
}

// Projects the hit point onto the ray traced sun shadow mask. Returns false when the hit is not
// visible on screen or lies on a different surface, in which case the caller traces shadow rays.
bool SampleRtSunVisibility(float3 position, float3 normal, out float visibility)
{
    const float4 clip = mul(gGi.viewProjection, float4(position + normal * 0.03, 1.0));
    if (clip.w <= 0.0)
    {
        return false;
    }

    // Camera projection flips y: negative NDC y is the top of the framebuffer.
    const float2 uv = float2(clip.x / clip.w * 0.5 + 0.5, clip.y / clip.w * 0.5 + 0.5);
    if (any(uv < 0.0) || any(uv > 1.0))
    {
        return false;
    }

    const int2 size = int2(gGi.screenSize.xy);
    const int2 pixel = clamp(int2(uv * gGi.screenSize.xy), int2(0, 0), size - 1);
    const float clipDepth = clip.z / clip.w;
    if (abs(gDepth.Load(int3(pixel, 0)).r - clipDepth) > max(0.01, clipDepth * 0.02))
    {
        return false;
    }

    float sum = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            sum += gRtSunShadow.Load(int3(clamp(pixel + int2(x, y), int2(0, 0), size - 1), 0));
        }
    }
    visibility = sum / 9.0;
    return true;
}

struct SurfaceSample
{
    float3 position;
    float3 normal;          // flipped toward the incoming ray
    float3 geometricNormal; // original winding
    float3 viewDirection;   // from the surface toward the ray origin
    float3 baseColor;
    float alpha;
    float roughness;
    float metalness;
    bool alphaMask;
    float alphaCutoff;
    bool alphaBlend;
    float transmission;
    float ior;
    float thickness;
    float dispersion;
    float3 attenuationColor;
    float attenuationDistance;
    float3 sheenColor;
    float sheenRoughness;
    float3 emissive;
};

SurfaceSample LoadSurface(
    uint geometryIndex,
    uint primitiveIndex,
    float2 barycentrics,
    float3 rayOrigin,
    float3 rayDirection,
    float hitDistance)
{
    const GpuGeometryData geometry = gGeometry[geometryIndex];
    const ByteAddressBuffer vertexBuffer = gBuffers[NonUniformResourceIndex(geometry.bufferIndices.x)];
    const ByteAddressBuffer indexBuffer = gBuffers[NonUniformResourceIndex(geometry.bufferIndices.y)];
    const uint stride = geometry.bufferIndices.z;

    const uint base = primitiveIndex * 12u;
    const uint i0 = indexBuffer.Load(base);
    const uint i1 = indexBuffer.Load(base + 4u);
    const uint i2 = indexBuffer.Load(base + 8u);

    const float w0 = 1.0 - barycentrics.x - barycentrics.y;
    const float3 n0 = LoadFloat3(vertexBuffer, i0 * stride + 12u);
    const float3 n1 = LoadFloat3(vertexBuffer, i1 * stride + 12u);
    const float3 n2 = LoadFloat3(vertexBuffer, i2 * stride + 12u);
    const float3 normalSum = n0 * w0 + n1 * barycentrics.x + n2 * barycentrics.y;
    const float normalLengthSquared = dot(normalSum, normalSum);
    // Degenerate normals would turn every downstream term into NaN; fall back to a normal facing
    // the ray origin.
    const float3 geometricNormal =
        normalLengthSquared > 1e-8 ? normalSum * rsqrt(normalLengthSquared) : -rayDirection;

    const float2 uv0 = LoadFloat2(vertexBuffer, i0 * stride + 24u);
    const float2 uv1 = LoadFloat2(vertexBuffer, i1 * stride + 24u);
    const float2 uv2 = LoadFloat2(vertexBuffer, i2 * stride + 24u);
    const float2 uv = uv0 * w0 + uv1 * barycentrics.x + uv2 * barycentrics.y;

    const float4 baseColorSample =
        gTextures[NonUniformResourceIndex(geometry.textureIndices.x)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.samplerIndices.x)],
            uv,
            0.0);
    const float transmissionSample =
        gTextures[NonUniformResourceIndex(geometry.volumeTextureIndices.x)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.volumeSamplerIndices.x)],
            uv,
            0.0).r;
    const float thicknessSample =
        gTextures[NonUniformResourceIndex(geometry.volumeTextureIndices.y)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.volumeSamplerIndices.y)],
            uv,
            0.0).g;
    const float3 sheenColorSample =
        gTextures[NonUniformResourceIndex(geometry.sheenTextureIndices.x)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.sheenSamplerIndices.x)],
            uv,
            0.0).rgb;
    const float sheenRoughnessSample =
        gTextures[NonUniformResourceIndex(geometry.sheenTextureIndices.y)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.sheenSamplerIndices.y)],
            uv,
            0.0).a;
    const float3 emissiveSample =
        gTextures[NonUniformResourceIndex(geometry.emissiveTextureIndices.x)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.emissiveSamplerIndices.x)],
            uv,
            0.0).rgb;

    SurfaceSample surface;
    surface.position = rayOrigin + rayDirection * hitDistance;
    surface.geometricNormal = geometricNormal;
    surface.viewDirection = -rayDirection;
    surface.normal = dot(geometricNormal, rayDirection) > 0.0 ? -geometricNormal : geometricNormal;
    surface.baseColor = baseColorSample.rgb * geometry.baseColorFactor.rgb;
    surface.alpha = baseColorSample.a * geometry.baseColorFactor.a;
    surface.roughness = clamp(geometry.params.x, 0.0, 1.0);
    surface.metalness = clamp(geometry.params.y, 0.0, 1.0);
    // params.z encodes the alpha mode: 0 opaque, 1 MASK, 2 BLEND (see GpuGeometryData).
    surface.alphaMask = abs(geometry.params.z - 1.0) < 0.5;
    surface.alphaCutoff = geometry.params.w;
    surface.alphaBlend = geometry.params.z > 1.5;
    surface.transmission = saturate(geometry.volume.x * transmissionSample);
    surface.ior = max(geometry.volume.y, 1.0);
    surface.thickness = max(geometry.volume.z * thicknessSample, 0.0);
    surface.dispersion = max(geometry.volume.w, 0.0);
    surface.attenuationColor = geometry.attenuation.rgb;
    surface.attenuationDistance = geometry.attenuation.a;
    surface.sheenColor = geometry.sheen.rgb * sheenColorSample;
    surface.sheenRoughness = saturate(geometry.sheen.a * sheenRoughnessSample);
    surface.emissive = geometry.emissive.rgb * emissiveSample;
    return surface;
}

bool TraceSceneRay(
    float3 origin,
    float3 direction,
    float maxDistance,
    out uint geometryIndex,
    out uint primitiveIndex,
    out float2 barycentrics,
    out float hitDistance)
{
    RayQuery<RAY_FLAG_NONE> query;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = maxDistance;
    query.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        return false;
    }

    geometryIndex = query.CommittedInstanceID();
    primitiveIndex = query.CommittedPrimitiveIndex();
    barycentrics = query.CommittedTriangleBarycentrics();
    hitDistance = query.CommittedRayT();
    return true;
}

// Windowed inverse-square falloff, matching the analytic lighting pass.
float DistanceAttenuation(float distance, float range)
{
    const float distanceSquared = max(distance * distance, 1e-4);
    const float ratio = distance / max(range, 1e-4);
    const float ratio2 = ratio * ratio;
    const float ratio4 = ratio2 * ratio2;
    const float cutoff = saturate(1.0 - ratio4);
    return cutoff * cutoff / distanceSquared;
}

// One occlusion ray per punctual light; returns 0 when the light is blocked.
float TraceLightVisibility(
    float3 position,
    float3 normal,
    float3 lightDirection,
    float lightDistance,
    float bias)
{
    const float offset = max(bias, 0.005);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    RayDesc ray;
    ray.Origin = position + normal * offset;
    ray.Direction = lightDirection;
    ray.TMin = 0.0;
    ray.TMax = max(lightDistance - offset * 2.0, 0.0);
    query.TraceRayInline(gTlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    query.Proceed();
    return query.CommittedStatus() != COMMITTED_TRIANGLE_HIT ? 1.0 : 0.0;
}

// Outgoing radiance of an opaque hit: the Cook-Torrance response to the sun and the punctual GI
// lights plus the sky/ambient term and the multi-bounce feedback from the previous frame.
float3 ShadeOpaqueSurface(
    const SurfaceSample surface,
    Texture2D feedbackTexture,
    uint2 pixel,
    uint frame)
{
    const float3 hitPosition = surface.position;
    const float3 normal = surface.normal;
    const float3 viewDirection = surface.viewDirection;
    const float3 f0 =
        lerp(float3(0.04, 0.04, 0.04), surface.baseColor, clamp(surface.metalness, 0.0, 1.0));
    const float roughness = clamp(surface.roughness, 0.04, 1.0);

    const float3 sunDirection = gGi.sunDirectionIntensity.xyz;
    const float sunNdotL = max(dot(normal, sunDirection), 0.0);
    const bool sunActive =
        gGi.params3.z > 0.5 && gGi.sunDirectionIntensity.w > 0.0 && sunNdotL > 0.0;

    // Sky/ambient stays diffuse; direct lights are shaded with the full BRDF. Emissive surfaces
    // add their emitted radiance regardless of lighting.
    float3 radiance = surface.baseColor * GiIrradianceOverPi(normal) + surface.emissive;

    if (sunActive)
    {
        float visibility = 0.0;
        // Prefer the ray traced sun mask: it is already smoothed by the lighting resolve and
        // removes the binary visibility noise from the indirect signal.
        if (gGi.params4.w <= 0.5 || !SampleRtSunVisibility(hitPosition, normal, visibility))
        {
            const uint sampleCount = uint(clamp(gGi.params4.x, 1.0, 16.0));
            const float angularRadius = max(gGi.params4.y, 0.0);
            const float3 coneTangent = normalize(
                cross(abs(sunDirection.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0),
                      sunDirection));
            const float3 coneBitangent = cross(sunDirection, coneTangent);
            const float3 shadowOrigin = hitPosition + normal * 0.02;

            for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
            {
                float3 direction = sunDirection;
                if (angularRadius > 0.0)
                {
                    const float radius =
                        angularRadius * sqrt(Hash01(pixel, frame, 101u + sampleIndex));
                    const float phi = 2.0 * kPi * Hash01(pixel, frame, 137u + sampleIndex);
                    direction = normalize(
                        sunDirection +
                        (coneTangent * cos(phi) + coneBitangent * sin(phi)) * radius);
                }

                RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
                RayDesc shadowRay;
                shadowRay.Origin = shadowOrigin;
                shadowRay.Direction = direction;
                shadowRay.TMin = 0.0;
                // The sun is effectively at infinity: the occlusion test must not be limited
                // by the indirect ray budget.
                shadowRay.TMax = 10000.0;
                shadowQuery.TraceRayInline(
                    gTlas,
                    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                    0xFF,
                    shadowRay);
                shadowQuery.Proceed();
                if (shadowQuery.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
                {
                    visibility += 1.0;
                }
            }

            visibility /= float(sampleCount);
        }

        const float3 sunRadiance =
            gGi.sunColor.rgb * gGi.sunDirectionIntensity.w * visibility;
        radiance += EvaluateCookTorrance(
            surface.baseColor,
            f0,
            surface.metalness,
            roughness,
            normal,
            viewDirection,
            sunDirection,
            sunRadiance);
        if (any(surface.sheenColor > 1e-5))
        {
            radiance += EvaluateSheenBRDF(
                surface.sheenColor,
                surface.sheenRoughness,
                normal,
                viewDirection,
                sunDirection,
                sunRadiance,
                sunNdotL);
        }
    }

    // Punctual GI lights (point and spot) with one ray traced shadow ray per light.
    const uint giLightCount = min(uint(max(gGi.params8.x, 0.0) + 0.5), kMaxGiLights);
    if (giLightCount > 0)
    {
        const bool lightShadows = gGi.params8.y > 0.5;
        const float shadowBias = max(gGi.params8.z, 0.0);
        for (uint lightIndex = 0; lightIndex < giLightCount; ++lightIndex)
        {
            const GpuLight light = gGiLights[lightIndex];
            const float3 offset = light.positionRange.xyz - hitPosition;
            const float distance = length(offset);
            const float3 lightDirection = offset / max(distance, 1e-4);
            const float attenuation = DistanceAttenuation(distance, light.positionRange.w);
            if (attenuation <= 1e-4)
            {
                continue;
            }

            float cone = 1.0;
            if (uint(light.params.x + 0.5) != 1u)
            {
                const float spotCosine = dot(-lightDirection, light.directionIntensity.xyz);
                cone = saturate(
                    (spotCosine - light.params.z) / max(light.params.y - light.params.z, 1e-4));
                cone *= cone;
                if (cone <= 0.0)
                {
                    continue;
                }
            }

            const float nDotL = dot(normal, lightDirection);
            if (nDotL <= 0.0)
            {
                continue;
            }

            float visibility = 1.0;
            if (lightShadows && light.params.w > 0.5)
            {
                visibility = TraceLightVisibility(
                    hitPosition,
                    normal,
                    lightDirection,
                    distance,
                    shadowBias);
            }

            const float3 lightRadiance =
                light.color.rgb * light.directionIntensity.w * attenuation * cone * visibility;
            radiance += EvaluateCookTorrance(
                surface.baseColor,
                f0,
                surface.metalness,
                roughness,
                normal,
                viewDirection,
                lightDirection,
                lightRadiance);
            if (any(surface.sheenColor > 1e-5))
            {
                radiance += EvaluateSheenBRDF(
                    surface.sheenColor,
                    surface.sheenRoughness,
                    normal,
                    viewDirection,
                    lightDirection,
                    lightRadiance,
                    saturate(nDotL));
            }
        }
    }

    const float4 previousClip = mul(gGi.previousViewProjection, float4(hitPosition, 1.0));
    if (previousClip.w > 0.0)
    {
        // Camera projection flips y: negative NDC y is the top of the framebuffer.
        const float2 previousUv = float2(
            previousClip.x / previousClip.w * 0.5 + 0.5,
            previousClip.y / previousClip.w * 0.5 + 0.5);
        if (all(previousUv >= 0.0) && all(previousUv <= 1.0))
        {
            const float3 feedback = SampleBilinear(feedbackTexture, previousUv);
            if (all(isfinite(feedback)))
            {
                radiance += clamp(feedback, 0.0, kMaxGiRadiance) * gGi.params.w;
            }
        }
    }
    return clamp(radiance, 0.0, kMaxGiRadiance);
}

// Traces the scene starting from the provided first hit, following alpha coverage and dielectric
// transmission (reflection/refraction, volume absorption). `iorScale` shifts the index of
// refraction for chromatic dispersion.
float3 TraceHitPath(
    Texture2D feedbackTexture,
    uint geometryIndex,
    uint primitiveIndex,
    float2 barycentrics,
    float3 rayOrigin,
    float3 rayDirection,
    float hitDistance,
    uint2 pixel,
    uint frame,
    float iorScale)
{
    float3 throughput = 1.0;
    float3 radiance = 0.0;

    uint currentGeometry = geometryIndex;
    uint currentPrimitive = primitiveIndex;
    float2 currentBarycentrics = barycentrics;
    float currentHitDistance = hitDistance;
    float3 origin = rayOrigin;
    float3 direction = rayDirection;
    const uint maxBounces = uint(clamp(gGi.params7.x, 1.0, 8.0));
    bool hasHit = true;

    for (uint bounce = 0; bounce <= maxBounces; ++bounce)
    {
        if (!hasHit)
        {
            radiance += throughput * GiSkyRadiance(direction);
            break;
        }

        const SurfaceSample surface = LoadSurface(
            currentGeometry,
            currentPrimitive,
            currentBarycentrics,
            origin,
            direction,
            currentHitDistance);

        const bool alphaRejected =
            (surface.alphaMask && surface.alpha < surface.alphaCutoff) ||
            (surface.alphaBlend && surface.alpha < 0.999 &&
             Hash01(pixel, frame, 211u + bounce) > surface.alpha);
        if (alphaRejected)
        {
            origin = surface.position + direction * 0.001;
            hasHit = TraceSceneRay(
                origin,
                direction,
                gGi.params.x,
                currentGeometry,
                currentPrimitive,
                currentBarycentrics,
                currentHitDistance);
            continue;
        }

        if (surface.transmission > 0.001)
        {
            const bool frontFace = dot(surface.geometricNormal, direction) < 0.0;
            const float3 normal = frontFace ? surface.geometricNormal : -surface.geometricNormal;
            const float ior = max(surface.ior * iorScale, 1.0);
            const float eta = frontFace ? (1.0 / ior) : ior;
            const float cosI = saturate(-dot(direction, normal));
            const float f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
            const float fresnel = f0 + (1.0 - f0) * pow(1.0 - cosI, 5.0);
            const float k = 1.0 - eta * eta * (1.0 - cosI * cosI);

            const float pick = Hash01(pixel, frame, 227u + bounce);
            if (k < 0.0 || pick < fresnel)
            {
                direction = reflect(direction, normal);
                origin = surface.position + normal * 0.001;
            }
            else
            {
                const float3 refracted = eta * direction + (eta * cosI - sqrt(max(k, 0.0))) * normal;
                const float refractedLengthSquared = dot(refracted, refracted);
                // Degenerate refraction would produce a NaN direction and fault the ray query;
                // fall back to a reflection instead.
                direction = refractedLengthSquared > 1e-8
                                ? refracted * rsqrt(refractedLengthSquared)
                                : reflect(direction, normal);
                origin = surface.position - normal * 0.001;
                if (surface.attenuationDistance > 0.0 && surface.thickness > 0.0)
                {
                    const float3 sigma =
                        -log(max(surface.attenuationColor, float3(1e-3, 1e-3, 1e-3))) /
                        surface.attenuationDistance;
                    throughput *= exp(-sigma * surface.thickness * max(gGi.params7.z, 0.0));
                }
            }

            hasHit = TraceSceneRay(
                origin,
                direction,
                gGi.params.x,
                currentGeometry,
                currentPrimitive,
                currentBarycentrics,
                currentHitDistance);
            continue;
        }

        radiance += throughput * ShadeOpaqueSurface(surface, feedbackTexture, pixel, frame);
        break;
    }

    return clamp(radiance, 0.0, kMaxGiRadiance);
}

float3 ShadeHit(
    Texture2D feedbackTexture,
    uint geometryIndex,
    uint primitiveIndex,
    float2 barycentrics,
    float3 rayOrigin,
    float3 rayDirection,
    float hitDistance,
    uint2 pixel,
    uint frame)
{
    const SurfaceSample first =
        LoadSurface(geometryIndex, primitiveIndex, barycentrics, rayOrigin, rayDirection, hitDistance);

    // Chromatic dispersion: trace one path per color channel with a shifted index of refraction.
    if (first.transmission > 0.001 && first.dispersion > 0.0 && gGi.params7.y > 0.5)
    {
        float3 dispersed = 0.0;
        for (int channel = 0; channel < 3; ++channel)
        {
            const float offset = (float(channel) - 1.0) * first.dispersion * 0.25;
            const float3 path = TraceHitPath(
                feedbackTexture,
                geometryIndex,
                primitiveIndex,
                barycentrics,
                rayOrigin,
                rayDirection,
                hitDistance,
                pixel,
                frame + uint(channel) * 5821u,
                1.0 + offset);
            dispersed[channel] = path[channel];
        }
        return dispersed;
    }

    return TraceHitPath(
        feedbackTexture,
        geometryIndex,
        primitiveIndex,
        barycentrics,
        rayOrigin,
        rayDirection,
        hitDistance,
        pixel,
        frame,
        1.0);
}
