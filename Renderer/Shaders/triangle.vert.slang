struct PushConstants
{
    float4 tint;
    float4 params;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct VertexInput
{
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float3 color : COLOR;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    const float angle = gPushConstants.params.x;
    const float sine = sin(angle);
    const float cosine = cos(angle);
    const float2 rotated = float2(
        input.position.x * cosine - input.position.y * sine,
        input.position.x * sine + input.position.y * cosine);

    VertexOutput output;
    output.position = float4(rotated, 0.0, 1.0);
    output.color = input.color * gPushConstants.tint.rgb;
    return output;
}
