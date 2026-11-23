// Flattened so the std430 block matches mukki::DeferredRenderer::ShadowPushConstants; Slang
// pads nested structs to 16 bytes, which would move viewIndex past the C++ layout.
struct ShadowPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params;
    uint2 textureIndices;
    uint2 samplerIndices;
    uint drawIndex;
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

// Must match mukki::DeferredRenderer::GpuObjectData (skinning drives the skin below).
struct GpuObjectData
{
    float4x4 previousModel;
    float4 specular;
    uint4 specularTextureIndices;
    uint4 specularSamplerIndices;
    float4 volume;
    float4 attenuation;
    uint4 volumeTextureIndices;
    uint4 volumeSamplerIndices;
    float4 alpha;
    float4 sheen;
    uint4 sheenTextureIndices;
    uint4 sheenSamplerIndices;
    float4 emissive;
    uint4 emissiveTextureIndices;
    uint4 emissiveSamplerIndices;
    float4 subsurface;
    uint4 subsurfaceTextureIndices;
    uint4 subsurfaceSamplerIndices;
    uint4 skinning; // x: joint offset, y: previous joint offset, z: joint count, w: 1 = skinned
};

// Must match mukki::DeferredRenderer::GpuMorphData.
struct GpuMorphData
{
    uint4 data;    // x: delta buffer bindless slot, y: target count, z: vertex count, w: unused
    uint4 offsets; // x: current weights offset, y: previous weights offset, z/w: unused
};

[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniform> gShadow;
[[vk::binding(1, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::binding(2, 0)]] StructuredBuffer<float4x4> gJoints;
[[vk::binding(3, 0)]] StructuredBuffer<GpuMorphData> gMorphData;
[[vk::binding(4, 0)]] StructuredBuffer<float4> gMorphWeights;
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];
[[vk::push_constant]] ConstantBuffer<ShadowPushConstants> gPush;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] uint4 joints : JOINTS0;
    [[vk::location(4)]] float4 weights : WEIGHTS0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input, uint vertexId : SV_VertexID)
{
    float4 localPosition = float4(input.position, 1.0);

    // Morph deltas apply in mesh space before skinning (current pose only; shadow views do not
    // need the previous pose).
    const GpuMorphData morph = gMorphData[gPush.drawIndex];
    if (morph.data.y != 0u)
    {
        for (uint target = 0; target < morph.data.y; ++target)
        {
            const float weight = gMorphWeights[morph.offsets.x + target].x;
            if (weight == 0.0f)
            {
                continue;
            }
            const ByteAddressBuffer buffer =
                gBuffers[NonUniformResourceIndex(morph.data.x)];
            const uint element = (target * morph.data.z + vertexId) * 8u;
            localPosition.xyz += float3(
                asfloat(buffer.Load(element)),
                asfloat(buffer.Load(element + 4u)),
                asfloat(buffer.Load(element + 8u))) * weight;
        }
    }

    const uint4 skinning = gObjects[gPush.drawIndex].skinning;
    if (skinning.w != 0u)
    {
        const float4x4 skinMatrix =
            gJoints[skinning.x + input.joints.x] * input.weights.x +
            gJoints[skinning.x + input.joints.y] * input.weights.y +
            gJoints[skinning.x + input.joints.z] * input.weights.z +
            gJoints[skinning.x + input.joints.w] * input.weights.w;
        localPosition = mul(skinMatrix, localPosition);
    }

    const float4 worldPosition = mul(gPush.model, localPosition);

    VertexOutput output;
    output.position = mul(gShadow.viewProjections[gPush.viewIndex], worldPosition);
    output.uv = input.uv;
    return output;
}
