// Compute meshlet culling for the G-buffer mesh shader path. One dispatch per draw culls its
// meshlets against the view frustum and the normal cone, compacts the surviving meshlet indices
// into the frame index buffer and accumulates the mesh-task group count read by the indirect draw.

#include "meshlet.slang"

// Must match mukki::SceneUniform / gbuffer.mesh.slang.
struct SceneUniform
{
    float4x4 viewProjection;
    float4x4 previousViewProjection;
    float4x4 view;
    float4x4 inverseViewProjection;
    float4 cameraPosition;
    float4 ambientColor;
    float4 clusterParams;
    uint4 clusterGrid;
    float4 screenSize;
    float4 reflectionParams;
    float4 sunDirection;
    float4 sunColor;
    float4 skySh[9];
    float4 skyParams;
    float4 skyZenith;
    float4 skyHorizon;
    float4 skyGround;
    float4 skyMap;
};

// Must match mukki::DeferredRenderer::MeshletCullPushConstants.
struct MeshletCullPushConstants
{
    float4x4 model;
    uint drawIndex;
    uint meshletCount;
};

// Mesh task commands are stored as three uints per draw (VkDrawMeshTasksIndirectCommandEXT).
static const uint kMeshTaskCommandUints = 3;

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::push_constant]] ConstantBuffer<MeshletCullPushConstants> gCull;
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
    if (CullMeshlet(gScene.viewProjection, gCull.model, gScene.cameraPosition.xyz, meshlet))
    {
        return;
    }

    uint slot = 0;
    InterlockedAdd(gMeshTaskCommands[gCull.drawIndex * kMeshTaskCommandUints], 1u, slot);
    gCompactedMeshlets[draw.compactedOffset + slot] = meshletIndex;
}
