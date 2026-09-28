struct ClusterUniform
{
    float4x4 inverseView;
    float4 cameraParams;  // x: near, y: far, z: tan(fovY / 2), w: aspect ratio
    float4 clusterParams; // x: tile size in pixels, y: z slice count
    uint4 grid;           // x, y, z grid dimensions, w: light count
    uint4 screen;         // x: width, y: height
};

struct GpuLight
{
    float4 positionRange;
    float4 directionIntensity;
    float4 color;
    float4 params; // x: type, y: inner cone cosine, z: outer cone cosine
};

[[vk::binding(0, 0)]] ConstantBuffer<ClusterUniform> gCluster;
[[vk::binding(1, 0)]] StructuredBuffer<GpuLight> gLights;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> gClusterCounts;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> gClusterIndices;

static const uint kMaxLightsPerCluster = 64;
static const uint kLightTypeDirectional = 0;

bool SphereIntersectsAabb(float3 center, float radius, float3 boundsMin, float3 boundsMax)
{
    const float3 closest = clamp(center, boundsMin, boundsMax);
    const float3 offset = center - closest;
    return dot(offset, offset) <= radius * radius;
}

// ndc is the Vulkan framebuffer convention: x and y in [-1, 1], y growing downward. depth is the
// positive view space distance from the camera.
float3 WorldPoint(float2 ndc, float depth)
{
    const float3 right = normalize(gCluster.inverseView[0].xyz);
    const float3 up = normalize(gCluster.inverseView[1].xyz);
    const float3 forward = -normalize(gCluster.inverseView[2].xyz);
    const float3 eye = gCluster.inverseView[3].xyz;

    const float x = ndc.x * gCluster.cameraParams.w * gCluster.cameraParams.z * depth;
    const float y = -ndc.y * gCluster.cameraParams.z * depth;
    return eye + right * x + up * y + forward * depth;
}

[numthreads(4, 4, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= gCluster.grid.x ||
        dispatchThreadId.y >= gCluster.grid.y ||
        dispatchThreadId.z >= gCluster.grid.z)
    {
        return;
    }

    const uint clusterIndex =
        (dispatchThreadId.z * gCluster.grid.y + dispatchThreadId.y) * gCluster.grid.x + dispatchThreadId.x;

    const float tileSize = gCluster.clusterParams.x;
    const float2 pixelMin = float2(dispatchThreadId.xy) * tileSize;
    const float2 pixelMax = min(pixelMin + tileSize, float2(gCluster.screen.xy));
    const float2 ndcMin = pixelMin / float2(gCluster.screen.xy) * 2.0 - 1.0;
    const float2 ndcMax = pixelMax / float2(gCluster.screen.xy) * 2.0 - 1.0;

    const float nearZ = gCluster.cameraParams.x;
    const float farZ = gCluster.cameraParams.y;
    const float sliceRatio = farZ / nearZ;
    const float nearSlice = nearZ * pow(sliceRatio, float(dispatchThreadId.z) / gCluster.clusterParams.y);
    const float farSlice = nearZ * pow(sliceRatio, float(dispatchThreadId.z + 1) / gCluster.clusterParams.y);

    const float3 corners[8] = {
        WorldPoint(float2(ndcMin.x, ndcMin.y), nearSlice),
        WorldPoint(float2(ndcMax.x, ndcMin.y), nearSlice),
        WorldPoint(float2(ndcMin.x, ndcMax.y), nearSlice),
        WorldPoint(float2(ndcMax.x, ndcMax.y), nearSlice),
        WorldPoint(float2(ndcMin.x, ndcMin.y), farSlice),
        WorldPoint(float2(ndcMax.x, ndcMin.y), farSlice),
        WorldPoint(float2(ndcMin.x, ndcMax.y), farSlice),
        WorldPoint(float2(ndcMax.x, ndcMax.y), farSlice),
    };

    float3 boundsMin = corners[0];
    float3 boundsMax = corners[0];
    [unroll]
    for (uint corner = 1; corner < 8; ++corner)
    {
        boundsMin = min(boundsMin, corners[corner]);
        boundsMax = max(boundsMax, corners[corner]);
    }

    const uint lightCount = gCluster.grid.w;
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        const GpuLight light = gLights[lightIndex];
        const uint type = uint(light.params.x);
        bool intersects = true;
        if (type != kLightTypeDirectional)
        {
            intersects = SphereIntersectsAabb(
                light.positionRange.xyz,
                light.positionRange.w,
                boundsMin,
                boundsMax);
        }

        if (intersects)
        {
            uint slot = 0;
            InterlockedAdd(gClusterCounts[clusterIndex], 1u, slot);
            if (slot < kMaxLightsPerCluster)
            {
                gClusterIndices[clusterIndex * kMaxLightsPerCluster + slot] = lightIndex;
            }
        }
    }
}
