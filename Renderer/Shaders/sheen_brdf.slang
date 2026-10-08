#ifndef MRE_SHEEN_BRDF
#define MRE_SHEEN_BRDF

// KHR_materials_sheen: Charlie distribution with Ashikhmin visibility (Estevez & Kulla), matching
// the glTF sample renderer. `nDotL` is the light cosine already computed by the caller.
float3 EvaluateSheenBRDF(
    float3 sheenColor,
    float sheenRoughness,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    float3 radiance,
    float nDotL)
{
    if (all(sheenColor <= float3(1e-5, 1e-5, 1e-5)) || nDotL <= 0.0)
    {
        return 0.0;
    }

    // Guard the degenerate retroreflection configuration (see brdf.slang).
    const float3 halfSum = viewDirection + lightDirection;
    const float halfSumLengthSquared = dot(halfSum, halfSum);
    const float3 halfVector =
        halfSumLengthSquared > 1e-8 ? halfSum * rsqrt(halfSumLengthSquared) : normal;
    const float nDotV = max(dot(normal, viewDirection), 1e-4);
    const float nDotH = max(dot(normal, halfVector), 0.0);

    const float alpha = max(sheenRoughness * sheenRoughness, 1e-4);
    const float invAlpha = 1.0 / alpha;
    const float cos2h = nDotH * nDotH;
    const float sin2h = max(1.0 - cos2h, 0.0);
    const float distribution =
        (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * 3.14159265359);
    const float visibility = 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-4);

    return sheenColor * distribution * visibility * radiance * nDotL;
}

#endif
