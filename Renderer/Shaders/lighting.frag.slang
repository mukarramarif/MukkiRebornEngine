struct SceneUniform
{
    float4x4 viewProjection;
    float4 cameraPosition;
    float4 lightDirection;
    float4 lightColor;
    float4 ambientColor;
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] Sampler2D gAlbedo;
[[vk::binding(2, 0)]] Sampler2D gNormal;
[[vk::binding(3, 0)]] Sampler2D gPosition;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

float3 acesTonemap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    const float4 albedoMetalness = gAlbedo.Sample(input.uv);
    const float4 normalRoughness = gNormal.Sample(input.uv);
    const float4 worldPosition = gPosition.Sample(input.uv);

    if (worldPosition.w < 0.5)
    {
        return float4(gScene.ambientColor.rgb, 1.0);
    }

    const float3 albedo = albedoMetalness.rgb;
    const float metalness = albedoMetalness.a;
    const float3 normal = normalize(normalRoughness.xyz * 2.0 - 1.0);
    const float roughness = normalRoughness.a;

    const float3 lightDirection = normalize(-gScene.lightDirection.xyz);
    const float3 viewDirection = normalize(gScene.cameraPosition.xyz - worldPosition.xyz);
    const float3 halfVector = normalize(lightDirection + viewDirection);

    const float ndotl = max(dot(normal, lightDirection), 0.0);
    const float ndoth = max(dot(normal, halfVector), 0.0);
    const float specularPower = exp2(11.0 * (1.0 - roughness) + 1.0);
    const float specular = pow(ndoth, specularPower);

    const float3 diffuseColor = albedo * (1.0 - metalness);
    const float3 specularColor = lerp(float3(0.04, 0.04, 0.04), albedo, metalness);
    const float3 radiance = gScene.lightColor.rgb * gScene.lightColor.w;

    float3 color = (diffuseColor + specularColor * specular) * radiance * ndotl;
    color += gScene.ambientColor.rgb * albedo;

    return float4(acesTonemap(color), 1.0);
}
