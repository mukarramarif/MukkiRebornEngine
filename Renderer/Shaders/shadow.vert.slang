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
    uint viewIndex;
};

// Must match mukki::kMaxShadowViews (4 cascades + 4 spots + 4 point lights * 6 faces).
struct ShadowUniform
{
    float4x4 viewProjections[32];
    float4 cascadeSplits;
    float4 params;   // x: depthBias, y: normalBias, z: distance, w: enabled
    float4 rtParams; // x: ray traced sun shadows enabled, y/z/w: unused
};

[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniform> gShadow;
[[vk::push_constant]] ConstantBuffer<ShadowPushConstants> gPush;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    const float4 worldPosition = mul(gPush.object.model, float4(input.position, 1.0));

    VertexOutput output;
    output.position = mul(gShadow.viewProjections[gPush.viewIndex], worldPosition);
    output.uv = input.uv;
    return output;
}
