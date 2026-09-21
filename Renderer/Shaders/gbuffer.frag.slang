struct ObjectPushConstants
{
    float4x4 model;
    float4 albedo;
    float4 params;
};

[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct GBufferOutput
{
    [[vk::location(0)]] float4 albedo : SV_Target0;
    [[vk::location(1)]] float4 normal : SV_Target1;
    [[vk::location(2)]] float4 position : SV_Target2;
};

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    GBufferOutput output;
    output.albedo = float4(gObject.albedo.rgb, gObject.params.y);
    output.normal = float4(normalize(input.normal) * 0.5 + 0.5, gObject.params.x);
    output.position = float4(input.worldPosition, 1.0);
    return output;
}
