// Flattened so the std430 block matches mukki::DeferredRenderer::ShadowPushConstants.
struct ShadowPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params; // x: roughness, y: metalness, z: alpha mask, w: alpha cutoff
    uint2 textureIndices;
    uint2 samplerIndices;
    uint drawIndex;
    uint viewIndex;
};

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];
[[vk::push_constant]] ConstantBuffer<ShadowPushConstants> gPush;

// Depth-only pass. Alpha-masked materials (Sponza's chain rails and foliage) still need their
// cutout so they cast shaped shadows instead of solid quads.
[shader("fragment")]
void fragmentMain([[vk::location(0)]] float2 uv : TEXCOORD0)
{
    if (gPush.params.z <= 0.5)
    {
        return;
    }

    const float4 baseColor = gTextures[NonUniformResourceIndex(gPush.textureIndices.x)].SampleLevel(
        gSamplers[NonUniformResourceIndex(gPush.samplerIndices.x)],
        uv,
        0.0);
    const float alpha = baseColor.a * gPush.baseColorFactor.a;
    if (alpha < gPush.params.w)
    {
        discard;
    }
}
