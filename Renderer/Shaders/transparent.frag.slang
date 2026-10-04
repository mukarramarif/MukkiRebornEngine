struct SceneUniform
{
    float4x4 viewProjection;
    float4x4 previousViewProjection;
    float4x4 view;
    float4 cameraPosition;
    float4 ambientColor;
    float4 clusterParams;
    uint4 clusterGrid;
    float4 screenSize;
    float4 reflectionParams; // x: reflection intensity
    float4 sunDirection;     // xyz: direction toward the sun
    float4 sunColor;         // rgb: color * intensity
};

struct ObjectPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params;
    uint2 textureIndices;
    uint2 samplerIndices;
    uint drawIndex;
};

// Must match DeferredRenderer::GpuObjectData.
struct GpuObjectData
{
    float4x4 previousModel;
    float4 specular;
    uint4 specularTextureIndices;
    uint4 specularSamplerIndices;
    float4 volume;
    float4 attenuation;
    uint4 volumeTextureIndices;
    uint4 volumeSamplerIndices;
    float4 alpha;
    float4 sheen;
    uint4 sheenTextureIndices;
    uint4 sheenSamplerIndices;
    float4 emissive; // rgb: emissiveFactor (incl. strength)
    uint4 emissiveTextureIndices;
    uint4 emissiveSamplerIndices;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 previousClip : PREVIOUSCLIP;
};

[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;
[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];

// Forward pass for glTF alphaMode = BLEND materials (windows, legacy glass). The G-buffer does not
// include these surfaces; they are alpha blended over the lit opaque image with the scene depth
// attached for occlusion. Shading is ambient plus the primary directional light.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    const float4 baseColorSample = gTextures[NonUniformResourceIndex(gObject.textureIndices.x)].Sample(
        gSamplers[NonUniformResourceIndex(gObject.samplerIndices.x)],
        input.uv);

    const float alpha = baseColorSample.a * gObject.baseColorFactor.a;
    if (alpha < 0.004)
    {
        discard;
    }

    const float3 albedo = baseColorSample.rgb * gObject.baseColorFactor.rgb;
    const float3 normal = normalize(input.normal);
    const float nDotL = max(dot(normal, -gScene.sunDirection.xyz), 0.0);
    const float3 lit = gScene.ambientColor.rgb + gScene.sunColor.rgb * nDotL;

    const GpuObjectData object = gObjects[gObject.drawIndex];
    const float3 emissiveSample =
        gTextures[NonUniformResourceIndex(object.emissiveTextureIndices.x)].Sample(
            gSamplers[NonUniformResourceIndex(object.emissiveSamplerIndices.x)],
            input.uv).rgb;
    const float3 emissive = object.emissive.rgb * emissiveSample;

    return float4(albedo * lit + emissive, alpha);
}
