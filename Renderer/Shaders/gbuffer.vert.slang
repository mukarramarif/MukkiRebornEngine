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
    float4 reflectionParams; // x: reflection intensity
    float4 sunDirection;     // xyz: direction toward the sun
    float4 sunColor;         // rgb: color * intensity
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
    float4 specular;               // x: specularFactor, yzw: specularColorFactor
    uint4 specularTextureIndices;  // x: specularTexture, y: specularColorTexture
    uint4 specularSamplerIndices;  // x: specular sampler, y: specular color sampler
    float4 volume;                 // x: transmission, y: ior, z: thickness, w: dispersion
    float4 attenuation;            // rgb: attenuation color, a: attenuation distance (0 = none)
    uint4 volumeTextureIndices;    // x: transmission texture, y: thickness texture
    uint4 volumeSamplerIndices;    // x: transmission sampler, y: thickness sampler
    float4 alpha;                  // x: blend, y: cutoff, z/w: unused
    float4 sheen;                  // rgb: sheenColorFactor, a: sheenRoughnessFactor
    uint4 sheenTextureIndices;     // x: sheenColorTexture, y: sheenRoughnessTexture
    uint4 sheenSamplerIndices;
    float4 emissive;               // rgb: emissiveFactor (incl. strength)
    uint4 emissiveTextureIndices;  // x: emissiveTexture
    uint4 emissiveSamplerIndices;  // x: emissive sampler
    float4 subsurface;            // rgb: diffuseTransmissionColor, a: factor
    uint4 subsurfaceTextureIndices; // x: diffuseTransmissionTexture, y: color texture
    uint4 subsurfaceSamplerIndices;
    uint4 skinning; // x: joint offset, y: previous joint offset, z: joint count, w: 1 = skinned
};

[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::binding(2, 0)]] StructuredBuffer<float4x4> gJoints;
[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] uint4 joints : JOINTS0;
    [[vk::location(4)]] float4 weights : WEIGHTS0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 previousClip : PREVIOUSCLIP;
    [[vk::location(4)]] float3 debugColor : MESHLETDEBUG;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    float4 localPosition = float4(input.position, 1.0);
    float4 previousLocalPosition = localPosition;
    float3 localNormal = input.normal;
    const uint4 skinning = gObjects[gObject.drawIndex].skinning;
    if (skinning.w != 0u)
    {
        const float4x4 skinMatrix =
            gJoints[skinning.x + input.joints.x] * input.weights.x +
            gJoints[skinning.x + input.joints.y] * input.weights.y +
            gJoints[skinning.x + input.joints.z] * input.weights.z +
            gJoints[skinning.x + input.joints.w] * input.weights.w;
        const float4x4 previousSkinMatrix =
            gJoints[skinning.y + input.joints.x] * input.weights.x +
            gJoints[skinning.y + input.joints.y] * input.weights.y +
            gJoints[skinning.y + input.joints.z] * input.weights.z +
            gJoints[skinning.y + input.joints.w] * input.weights.w;
        localPosition = mul(skinMatrix, localPosition);
        previousLocalPosition = mul(previousSkinMatrix, previousLocalPosition);
        localNormal = mul((float3x3)skinMatrix, localNormal);
    }

    const float4 worldPosition = mul(gObject.model, localPosition);
    const float4 previousWorldPosition =
        mul(gObjects[gObject.drawIndex].previousModel, previousLocalPosition);

    VertexOutput output;
    output.position = mul(gScene.viewProjection, worldPosition);
    output.worldPosition = worldPosition.xyz;
    output.normal = mul((float3x3)gObject.model, localNormal);
    output.uv = input.uv;
    output.previousClip = mul(gScene.previousViewProjection, previousWorldPosition);
    // Vertex-pipeline geometry is not split into meshlets; keep it dark in the debug view.
    output.debugColor = float3(0.08, 0.08, 0.08);
    return output;
}
