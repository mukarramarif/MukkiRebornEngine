struct ObjectPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params;
    uint2 textureIndices;
    uint2 samplerIndices;
};

struct ShadowPushConstants
{
    ObjectPushConstants object;
    uint cascadeIndex;
};

struct ShadowUniform
{
    float4x4 lightViewProjection[4];
    float4 cascadeSplits;
    float4 params; // x: depthBias, y: normalBias, z: distance, w: enabled
};

[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniform> gShadow;
[[vk::push_constant]] ConstantBuffer<ShadowPushConstants> gPush;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
};

[shader("vertex")]
float4 vertexMain(VertexInput input) : SV_Position
{
    const float4 worldPosition = mul(gPush.object.model, float4(input.position, 1.0));
    return mul(gShadow.lightViewProjection[gPush.cascadeIndex], worldPosition);
}
