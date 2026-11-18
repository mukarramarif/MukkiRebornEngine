// Mesh shader variant of the shadow pass. Alpha-masked materials still go through the same
// fragment stage, so it only emits position and uv.

#include "meshlet.slang"

// Flattened so the std430 block matches mukki::DeferredRenderer::ShadowPushConstants.
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

[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniform> gShadow;
[[vk::push_constant]] ConstantBuffer<ShadowPushConstants> gPush;
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];
[[vk::binding(0, 2)]] StructuredBuffer<GpuMeshDraw> gMeshDraws;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void meshMain(
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID,
    out vertices VertexOutput verts[64],
    out indices uint3 tris[124])
{
    const GpuMeshDraw draw = gMeshDraws[gPush.drawIndex];
    if (groupID.x >= draw.meshletCount)
    {
        SetMeshOutputCounts(0, 0);
        return;
    }

    const ByteAddressBuffer meshletBuffer = gBuffers[NonUniformResourceIndex(draw.meshletBufferIndex)];
    const GpuMeshlet meshlet = LoadMeshlet(meshletBuffer, draw.meshletOffset + groupID.x);
    const float4x4 viewProjection = gShadow.viewProjections[gPush.viewIndex];

    const float3 worldCenter = mul(gPush.model, float4(meshlet.centerRadius.xyz, 1.0)).xyz;
    const float3 axisScale = float3(
        length(gPush.model[0].xyz),
        length(gPush.model[1].xyz),
        length(gPush.model[2].xyz));
    const float worldRadius = meshlet.centerRadius.w * max(axisScale.x, max(axisScale.y, axisScale.z));
    if (IsMeshletOutsideFrustum(viewProjection, worldCenter, worldRadius))
    {
        SetMeshOutputCounts(0, 0);
        return;
    }

    const uint vertexCount = meshlet.data.x;
    const uint triangleCount = meshlet.data.y;
    SetMeshOutputCounts(vertexCount, triangleCount);

    const ByteAddressBuffer vertexBuffer = gBuffers[NonUniformResourceIndex(draw.vertexBufferIndex)];
    for (uint vertexIndex = groupThreadID.x; vertexIndex < vertexCount; vertexIndex += 32u)
    {
        const uint sourceVertex =
            meshletBuffer.Load(draw.vertexIndexBase + (meshlet.data.z + vertexIndex) * 4u);
        const MeshletVertex vertex = LoadMeshletVertex(vertexBuffer, sourceVertex);

        VertexOutput output;
        output.position = mul(viewProjection, mul(gPush.model, float4(vertex.position, 1.0)));
        output.uv = vertex.uv;
        verts[vertexIndex] = output;
    }

    for (uint triangle = groupThreadID.x; triangle < triangleCount; triangle += 32u)
    {
        const uint packed = meshletBuffer.Load(draw.triangleBase + (meshlet.data.w + triangle) * 4u);
        tris[triangle] = uint3(packed & 0xffu, (packed >> 8u) & 0xffu, (packed >> 16u) & 0xffu);
    }
}
