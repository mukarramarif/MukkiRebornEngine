// Mesh shader variant of the G-buffer pass. Each workgroup decodes one meshlet, culls it against
// the view frustum and the normal cone, then emits its deduplicated vertices and triangles.

#include "meshlet.slang"

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

struct ObjectPushConstants
{
    float4x4 model;
    float4 baseColorFactor;
    float4 params;
    uint2 textureIndices;
    uint2 samplerIndices;
    uint drawIndex;
};

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
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;
[[vk::binding(2, 1)]] ByteAddressBuffer gBuffers[];
[[vk::binding(0, 2)]] StructuredBuffer<GpuMeshDraw> gMeshDraws;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 previousClip : PREVIOUSCLIP;
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
    const GpuMeshDraw draw = gMeshDraws[gObject.drawIndex];
    if (groupID.x >= draw.meshletCount)
    {
        SetMeshOutputCounts(0, 0);
        return;
    }

    const ByteAddressBuffer meshletBuffer = gBuffers[NonUniformResourceIndex(draw.meshletBufferIndex)];
    const GpuMeshlet meshlet = LoadMeshlet(meshletBuffer, draw.meshletOffset + groupID.x);
    if (CullMeshlet(gScene.viewProjection, gObject.model, gScene.cameraPosition.xyz, meshlet))
    {
        SetMeshOutputCounts(0, 0);
        return;
    }

    const uint vertexCount = meshlet.data.x;
    const uint triangleCount = meshlet.data.y;
    SetMeshOutputCounts(vertexCount, triangleCount);

    const ByteAddressBuffer vertexBuffer = gBuffers[NonUniformResourceIndex(draw.vertexBufferIndex)];
    const float4x4 previousModel = gObjects[gObject.drawIndex].previousModel;

    for (uint vertexIndex = groupThreadID.x; vertexIndex < vertexCount; vertexIndex += 32u)
    {
        const uint sourceVertex =
            meshletBuffer.Load(draw.vertexIndexBase + (meshlet.data.z + vertexIndex) * 4u);
        const MeshletVertex vertex = LoadMeshletVertex(vertexBuffer, sourceVertex);

        const float4 localPosition = float4(vertex.position, 1.0);
        const float4 worldPosition = mul(gObject.model, localPosition);
        const float4 previousWorldPosition = mul(previousModel, localPosition);

        VertexOutput output;
        output.position = mul(gScene.viewProjection, worldPosition);
        output.worldPosition = worldPosition.xyz;
        output.normal = mul((float3x3)gObject.model, vertex.normal);
        output.uv = vertex.uv;
        output.previousClip = mul(gScene.previousViewProjection, previousWorldPosition);
        verts[vertexIndex] = output;
    }

    for (uint triangle = groupThreadID.x; triangle < triangleCount; triangle += 32u)
    {
        const uint packed = meshletBuffer.Load(draw.triangleBase + (meshlet.data.w + triangle) * 4u);
        tris[triangle] = uint3(packed & 0xffu, (packed >> 8u) & 0xffu, (packed >> 16u) & 0xffu);
    }
}
