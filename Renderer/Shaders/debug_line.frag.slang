struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
};

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return input.color;
}
