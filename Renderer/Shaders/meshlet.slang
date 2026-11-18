// Shared meshlet decoding and culling for the mesh shader pipelines. The includer must declare
// the uniform and push constant bindings before including this file.

// GltfVertex: position (12 bytes), normal (12 bytes), uv (8 bytes).
static const uint kMeshletVertexStride = 32;
static const uint kMeshletStride = 48;

struct GpuMeshlet
{
    float4 centerRadius;   // xyz: local space center, w: radius
    float4 coneAxisCutoff; // xyz: local cone axis, w: sin(half angle) (meshoptimizer convention)
    // x: vertex count, y: triangle count, z: vertex list offset, w: triangle list offset
    uint4 data;
};

// Must match mukki::DeferredRenderer::GpuMeshDraw.
struct GpuMeshDraw
{
    uint meshletOffset;      // first meshlet of the draw, in GpuMeshlet units
    uint meshletCount;
    uint vertexIndexBase;    // byte offset of the meshlet vertex index list
    uint triangleBase;       // byte offset of the packed micro triangle list
    uint vertexBufferIndex;  // bindless slot of the source GltfVertex buffer
    uint meshletBufferIndex; // bindless slot of the meshlet data buffer
    uint padding0;
    uint padding1;
};

float MeshletLoadFloat(ByteAddressBuffer buffer, uint byteOffset)
{
    return asfloat(buffer.Load(byteOffset));
}

GpuMeshlet LoadMeshlet(ByteAddressBuffer buffer, uint meshletIndex)
{
    const uint base = meshletIndex * kMeshletStride;
    GpuMeshlet meshlet;
    meshlet.centerRadius = float4(
        MeshletLoadFloat(buffer, base),
        MeshletLoadFloat(buffer, base + 4u),
        MeshletLoadFloat(buffer, base + 8u),
        MeshletLoadFloat(buffer, base + 12u));
    meshlet.coneAxisCutoff = float4(
        MeshletLoadFloat(buffer, base + 16u),
        MeshletLoadFloat(buffer, base + 20u),
        MeshletLoadFloat(buffer, base + 24u),
        MeshletLoadFloat(buffer, base + 28u));
    meshlet.data = uint4(
        buffer.Load(base + 32u),
        buffer.Load(base + 36u),
        buffer.Load(base + 40u),
        buffer.Load(base + 44u));
    return meshlet;
}

struct MeshletVertex
{
    float3 position;
    float3 normal;
    float2 uv;
};

MeshletVertex LoadMeshletVertex(ByteAddressBuffer buffer, uint vertexIndex)
{
    const uint base = vertexIndex * kMeshletVertexStride;
    MeshletVertex vertex;
    vertex.position = float3(
        MeshletLoadFloat(buffer, base),
        MeshletLoadFloat(buffer, base + 4u),
        MeshletLoadFloat(buffer, base + 8u));
    vertex.normal = float3(
        MeshletLoadFloat(buffer, base + 12u),
        MeshletLoadFloat(buffer, base + 16u),
        MeshletLoadFloat(buffer, base + 20u));
    vertex.uv = float2(
        MeshletLoadFloat(buffer, base + 24u),
        MeshletLoadFloat(buffer, base + 28u));
    return vertex;
}

// Sphere/frustum culling against the six clip planes derived from the view projection. Clip depth
// follows the engine convention (Vulkan style z in [0, 1]).
bool IsMeshletOutsideFrustum(float4x4 viewProjection, float3 center, float radius)
{
    const float4 planes[6] = {
        viewProjection[3] + viewProjection[0],
        viewProjection[3] - viewProjection[0],
        viewProjection[3] + viewProjection[1],
        viewProjection[3] - viewProjection[1],
        viewProjection[2],
        viewProjection[3] - viewProjection[2]};

    for (int index = 0; index < 6; ++index)
    {
        const float3 normal = planes[index].xyz;
        const float planeLength = length(normal);
        if (dot(normal, center) + planes[index].w < -radius * planeLength)
        {
            return true;
        }
    }
    return false;
}

bool CullMeshlet(
    float4x4 viewProjection,
    float4x4 model,
    float3 cameraPosition,
    GpuMeshlet meshlet)
{
    const float3 worldCenter = mul(model, float4(meshlet.centerRadius.xyz, 1.0)).xyz;
    const float3 axisScale = float3(length(model[0].xyz), length(model[1].xyz), length(model[2].xyz));
    const float worldRadius = meshlet.centerRadius.w * max(axisScale.x, max(axisScale.y, axisScale.z));

    if (IsMeshletOutsideFrustum(viewProjection, worldCenter, worldRadius))
    {
        return true;
    }

    // The cone test requires a non-mirrored transform (mirrored instances flip the winding).
    const float determinant = dot(cross(model[0].xyz, model[1].xyz), model[2].xyz);
    if (determinant <= 0.0)
    {
        return false;
    }

    // meshoptimizer backface test based on the bounding sphere (no cone apex):
    //   dot(normalize(center - camera), axis) >= cutoff + radius / distance
    // cutoff == 1 or a zero axis (degenerate cones) can never satisfy this and disable culling.
    const float3 coneAxis = mul((float3x3)model, meshlet.coneAxisCutoff.xyz);
    const float axisLength = length(coneAxis);
    if (axisLength <= 0.0)
    {
        return false;
    }

    const float3 axis = coneAxis / axisLength;
    const float3 toCenter = worldCenter - cameraPosition;
    const float distance = length(toCenter);
    return dot(toCenter, axis) >= meshlet.coneAxisCutoff.w * distance + worldRadius;
}
