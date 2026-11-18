// Compute meshlet culling for the shadow mesh shader views. One dispatch per (view, draw) culls
// the draw's meshlets against the shadow view's frustum, compacts the surviving meshlet indices
// and accumulates the indirect mesh-task group count for that view.

#include "meshlet.slang"

// Must match mukki::DeferredRenderer::ShadowUniform / shadow.mesh.slang.
struct ShadowUniform
{
    float4x4 viewProjections[32];
    float4 cascadeSplits;
    float4 params;
    float4 rtParams;
};

// Must match mukki::DeferredRenderer::ShadowCullPushConstants.
struct ShadowCullPushConstants
{
    float4x4 model;
    uint viewIndex;
    uint drawIndex;
    uint compactedOffset;
    uint commandIndex;
    uint meshletCount;
};

// Mesh task commands are stored as three uints per draw (VkDrawMeshTasksIndirectCommandEXT).
static const uint kMeshTaskCommandUints = 3;

[[vk::binding(0, 0)]] ConstantBuffer<ShadowUniform> gShadow;
[[vk::push_constant]] ConstantBuffer<ShadowCullPushConstants> gCull;
[[vk::binding(0, 1)]] StructuredBuffer<GpuMeshDraw> gMeshDraws;
[[vk::binding(1, 1)]] RWStructuredBuffer<uint> gCompactedMeshlets;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> gMeshTaskCommands;
[[vk::binding(2, 2)]] ByteAddressBuffer gBuffers[];

[numthreads(64, 1, 1)]
void computeMain(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
    const uint meshletIndex = groupID.x * 64u + groupThreadID.x;
    if (meshletIndex >= gCull.meshletCount)
    {
        return;
    }

    const GpuMeshDraw draw = gMeshDraws[gCull.drawIndex];
    const ByteAddressBuffer meshletBuffer = gBuffers[NonUniformResourceIndex(draw.meshletBufferIndex)];
    const GpuMeshlet meshlet = LoadMeshlet(meshletBuffer, draw.meshletOffset + meshletIndex);

    // Shadow views only frustum-cull (the shadow mesh shader has no normal cone test).
    const float3 worldCenter = mul(gCull.model, float4(meshlet.centerRadius.xyz, 1.0)).xyz;
    const float3 axisScale = float3(
        length(gCull.model[0].xyz),
        length(gCull.model[1].xyz),
        length(gCull.model[2].xyz));
    const float worldRadius =
        meshlet.centerRadius.w * max(axisScale.x, max(axisScale.y, axisScale.z));
    if (IsMeshletOutsideFrustum(
            gShadow.viewProjections[gCull.viewIndex], worldCenter, worldRadius))
    {
        return;
    }

    uint slot = 0;
    InterlockedAdd(gMeshTaskCommands[gCull.commandIndex * kMeshTaskCommandUints], 1u, slot);
    gCompactedMeshlets[gCull.compactedOffset + slot] = meshletIndex;
}
