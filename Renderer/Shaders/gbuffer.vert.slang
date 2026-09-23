struct SceneUniform
{
    float4x4 viewProjection;
    float4 cameraPosition;
    float4 lightDirection;
    float4 lightColor;
    float4 ambientColor;
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;

struct ObjectPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params;
    uint2 textureIndices;
    uint2 samplerIndices;
};

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
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    const float4 worldPosition = mul(gObject.model, float4(input.position, 1.0));

    VertexOutput output;
    output.position = mul(gScene.viewProjection, worldPosition);
    output.worldPosition = worldPosition.xyz;
    output.normal = mul((float3x3)gObject.model, input.normal);
    output.uv = input.uv;
    return output;
}
