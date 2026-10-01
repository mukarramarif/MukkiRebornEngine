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

[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;
[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 previousClip : PREVIOUSCLIP;
};

struct GBufferOutput
{
    [[vk::location(0)]] float4 albedo : SV_Target0;
    [[vk::location(1)]] float4 normal : SV_Target1;
    [[vk::location(2)]] float4 position : SV_Target2;
    [[vk::location(3)]] float2 motion : SV_Target3;
};

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    const float4 baseColorSample = gTextures[NonUniformResourceIndex(gObject.textureIndices.x)].Sample(
        gSamplers[NonUniformResourceIndex(gObject.samplerIndices.x)],
        input.uv);

    // Alpha masking: if the alpha channel of the base color texture is below the cutoff value, discard the fragment.
    if (gObject.params.z > 0.5)
    {
        const float alpha = baseColorSample.a * gObject.baseColorFactor.a;
        if (alpha < gObject.params.w)
        {
            discard;
        }
    }

    const float4 metalRoughness = gTextures[NonUniformResourceIndex(gObject.textureIndices.y)].Sample(
        gSamplers[NonUniformResourceIndex(gObject.samplerIndices.y)],
        input.uv);

    // Motion vectors in pixels: current pixel position minus the position where this surface was
    // on the previous frame (uv y grows downward, matching the framebuffer).
    const float2 previousNdc = input.previousClip.xy / input.previousClip.w;
    const float2 previousPixel = float2(
        (previousNdc.x * 0.5 + 0.5) * gScene.screenSize.x,
        (0.5 - previousNdc.y * 0.5) * gScene.screenSize.y);

    GBufferOutput output;
    output.albedo = float4(baseColorSample.rgb * gObject.baseColorFactor.rgb, gObject.params.y * metalRoughness.b);
    output.normal = float4(normalize(input.normal) * 0.5 + 0.5, gObject.params.x * metalRoughness.g);
    output.position = float4(input.worldPosition, 1.0);
    output.motion = input.position.xy - previousPixel;
    return output;
}
