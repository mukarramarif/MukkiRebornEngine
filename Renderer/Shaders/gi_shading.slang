// Shared hit shading used by the per-pixel GI trace and the probe update pass. The includer must
// declare GiUniform, GpuGeometryData and the bindings for gGi, gTlas, gGeometry, gDepth,
// gTextures, gSamplers, gBuffers and gRtSunShadow before including this file.

static const float kPi = 3.14159265359;
static const float kMaxGiRadiance = 20.0;

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

float3 SampleGgxDirection(float3 normal, float3 viewDirection, float roughness, uint2 pixel, uint frame)
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

    const float3 halfVector =
        tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)) + view * cosTheta;
    return reflect(-view, normalize(halfVector));
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

    const float2 uv = float2(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
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

// Outgoing radiance leaving the hit surface: albedo times (sun visibility plus sky) plus the
// multi-bounce feedback from the previous frame. `feedbackTexture` holds the previous indirect
// result; probe rays pass the white texture because they do not use screen-space feedback.
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
    float3 normal = normalize(n0 * w0 + n1 * barycentrics.x + n2 * barycentrics.y);
    if (dot(normal, rayDirection) > 0.0)
    {
        normal = -normal;
    }

    const float2 uv0 = LoadFloat2(vertexBuffer, i0 * stride + 24u);
    const float2 uv1 = LoadFloat2(vertexBuffer, i1 * stride + 24u);
    const float2 uv2 = LoadFloat2(vertexBuffer, i2 * stride + 24u);
    const float2 uv = uv0 * w0 + uv1 * barycentrics.x + uv2 * barycentrics.y;

    const float3 baseColor =
        gTextures[NonUniformResourceIndex(geometry.textureIndices.x)].SampleLevel(
            gSamplers[NonUniformResourceIndex(geometry.samplerIndices.x)],
            uv,
            0.0).rgb *
        geometry.baseColorFactor.rgb;

    const float3 hitPosition = rayOrigin + rayDirection * hitDistance;

    float3 incident = gGi.skyColor.rgb * gGi.skyColor.w;
    if (gGi.params3.z > 0.5 && gGi.sunDirectionIntensity.w > 0.0)
    {
        const float3 sunDirection = gGi.sunDirectionIntensity.xyz;
        const float nDotL = max(dot(normal, sunDirection), 0.0);
        if (nDotL > 0.0)
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
                const float3 shadowOrigin = hitPosition + normal * max(0.02, hitDistance * 0.005);

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

            incident += gGi.sunColor.rgb * gGi.sunDirectionIntensity.w * nDotL * visibility;
        }
    }

    float3 radiance = baseColor * incident;

    const float4 previousClip = mul(gGi.previousViewProjection, float4(hitPosition, 1.0));
    if (previousClip.w > 0.0)
    {
        const float2 previousUv = float2(
            previousClip.x / previousClip.w * 0.5 + 0.5,
            0.5 - previousClip.y / previousClip.w * 0.5);
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
