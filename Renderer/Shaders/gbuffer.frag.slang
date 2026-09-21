struct ObjectPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params;
};

[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;
[[vk::binding(0, 1)]] Sampler2D gBaseColorTexture;
[[vk::binding(1, 1)]] Sampler2D gMetalRoughnessTexture;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
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
    const float4 baseColorSample = gBaseColorTexture.Sample(input.uv);
    const float4 metalRoughness = gMetalRoughnessTexture.Sample(input.uv);

    GBufferOutput output;
    output.albedo = float4(baseColorSample.rgb * gObject.baseColorFactor.rgb, gObject.params.y * metalRoughness.b);
    output.normal = float4(normalize(input.normal) * 0.5 + 0.5, gObject.params.x * metalRoughness.g);
    output.position = float4(input.worldPosition, 1.0);
    return output;
}
