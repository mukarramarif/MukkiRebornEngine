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

struct GpuObjectData
{
    float4x4 previousModel;
    float4 specular;               // x: specularFactor, yzw: specularColorFactor
    uint4 specularTextureIndices;  // x: specularTexture, y: specularColorTexture
    uint4 specularSamplerIndices;  // x: specular sampler, y: specular color sampler
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 previousClip : PREVIOUSCLIP;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    const float4 localPosition = float4(input.position, 1.0);
    const float4 worldPosition = mul(gObject.model, localPosition);
    const float4 previousWorldPosition = mul(gObjects[gObject.drawIndex].previousModel, localPosition);

    VertexOutput output;
    output.position = mul(gScene.viewProjection, worldPosition);
    output.worldPosition = worldPosition.xyz;
    output.normal = mul((float3x3)gObject.model, input.normal);
    output.uv = input.uv;
    output.previousClip = mul(gScene.previousViewProjection, previousWorldPosition);
    return output;
}
