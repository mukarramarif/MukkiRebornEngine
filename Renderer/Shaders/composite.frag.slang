// Fullscreen composite: adds the diffused subsurface term to the offscreen HDR lighting image and
// applies ACES tone mapping. Debug views (cluster heatmap / GI only) bypass the tone mapping because
// the lighting pass already wrote display-referred values for them.

#include "tonemap.slang"

struct CompositeUniform
{
    // x: subsurface strength, y: debug view (0 off, 1 blurred, 2 sharp source), z: passthrough
    // (debug views), w: unused.
    float4 params;
};

[[vk::push_constant]] ConstantBuffer<CompositeUniform> gComposite;
[[vk::binding(0, 0)]] Texture2D gLighting;
[[vk::binding(1, 0)]] Texture2D gSssSource;
[[vk::binding(2, 0)]] Texture2D gSssBlurred;
[[vk::binding(3, 0)]] SamplerState gSampler;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    if (gComposite.params.y > 1.5)
    {
        return float4(
            acesTonemap(gSssSource.Sample(gSampler, input.uv).rgb * gComposite.params.x),
            1.0);
    }
    if (gComposite.params.y > 0.5)
    {
        return float4(
            acesTonemap(gSssBlurred.Sample(gSampler, input.uv).rgb * gComposite.params.x),
            1.0);
    }

    const float3 color =
        gLighting.Sample(gSampler, input.uv).rgb +
        gSssBlurred.Sample(gSampler, input.uv).rgb * gComposite.params.x;
    if (gComposite.params.z > 0.5)
    {
        return float4(color, 1.0);
    }
    return float4(acesTonemap(color), 1.0);
}
