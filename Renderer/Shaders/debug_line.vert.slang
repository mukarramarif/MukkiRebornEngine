struct DebugPushConstants
{
    float4x4 viewProjection;
};

[[vk::push_constant]] ConstantBuffer<DebugPushConstants> gPush;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float4 color : COLOR;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = mul(gPush.viewProjection, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}
