// Shared Cook-Torrance BRDF used by the analytic lighting pass and the ray traced hit shading.
// Kept in one place so direct lighting and indirect bounces stay consistent.
static const float kBrdfPi = 3.14159265359;

float3 BrdfFresnelSchlick(float cosTheta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float BrdfDistributionGgx(float nDotH, float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(kBrdfPi * denominator * denominator, 1e-6);
}

float BrdfGeometrySchlickGgx(float nDotX, float roughness)
{
    const float r = roughness + 1.0;
    const float k = (r * r) / 8.0;
    return nDotX / max(nDotX * (1.0 - k) + k, 1e-4);
}

// Returns the outgoing radiance for `radiance` arriving from `lightDirection`.
float3 EvaluateCookTorrance(
    float3 albedo,
    float3 f0,
    float metalness,
    float roughness,
    float3 normal,
    float3 viewDirection,
    float3 lightDirection,
    float3 radiance)
{
    const float nDotL = dot(normal, lightDirection);
    if (nDotL <= 0.0)
    {
        return 0.0;
    }

    const float3 halfVector = normalize(viewDirection + lightDirection);
    const float nDotV = max(dot(normal, viewDirection), 1e-4);
    const float nDotH = max(dot(normal, halfVector), 0.0);
    const float vDotH = max(dot(viewDirection, halfVector), 0.0);

    const float3 fresnel = BrdfFresnelSchlick(vDotH, f0);
    const float distribution = BrdfDistributionGgx(nDotH, roughness);
    const float geometry =
        BrdfGeometrySchlickGgx(nDotV, roughness) * BrdfGeometrySchlickGgx(nDotL, roughness);
    const float3 specular = (distribution * geometry * fresnel) / max(4.0 * nDotV * nDotL, 1e-4);

    const float3 diffuseWeight = (1.0 - fresnel) * (1.0 - metalness);
    return (diffuseWeight * albedo / kBrdfPi + specular) * radiance * nDotL;
}
