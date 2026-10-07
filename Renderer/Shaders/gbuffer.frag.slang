struct SceneUniform
{
    float4x4 viewProjection;
    float4x4 previousViewProjection;
    float4x4 view;
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
};

[[vk::push_constant]] ConstantBuffer<ObjectPushConstants> gObject;
[[vk::binding(0, 0)]] ConstantBuffer<SceneUniform> gScene;
[[vk::binding(1, 0)]] StructuredBuffer<GpuObjectData> gObjects;
[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[16];

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 worldPosition : WORLDPOS;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 previousClip : PREVIOUSCLIP;
};

struct GBufferOutput
{
    [[vk::location(0)]] float4 albedo : SV_Target0;
    [[vk::location(1)]] float4 normal : SV_Target1;
    [[vk::location(2)]] float4 position : SV_Target2; // w: object index + 1 (0 = sky)
    [[vk::location(3)]] float4 motion : SV_Target3;   // xy: motion, zw: object UV
    [[vk::location(4)]] float4 specularF0 : SV_Target4;
    [[vk::location(5)]] float4 transmission : SV_Target5;
    [[vk::location(6)]] float4 attenuation : SV_Target6;
    [[vk::location(7)]] float4 sheen : SV_Target7;
};

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    const float4 baseColorSample = gTextures[NonUniformResourceIndex(gObject.textureIndices.x)].Sample(
        gSamplers[NonUniformResourceIndex(gObject.samplerIndices.x)],
        input.uv);

    // Alpha masking: if the alpha channel of the base color texture is below the cutoff value, discard the fragment.
    if (gObject.params.z > 0.5)
    {
        const float alpha = baseColorSample.a * gObject.baseColorFactor.a;
        if (alpha < gObject.params.w)
        {
            discard;
        }
    }

    const float4 metalRoughness = gTextures[NonUniformResourceIndex(gObject.textureIndices.y)].Sample(
        gSamplers[NonUniformResourceIndex(gObject.samplerIndices.y)],
        input.uv);

    // Motion vectors in pixels: current pixel position minus the position where this surface was
    // on the previous frame. The camera projection flips y so negative NDC y is the top of the
    // framebuffer (Vulkan); NDC -> pixel must therefore not negate y.
    const float2 previousNdc = input.previousClip.xy / input.previousClip.w;
    const float2 previousPixel = float2(
        (previousNdc.x * 0.5 + 0.5) * gScene.screenSize.x,
        (previousNdc.y * 0.5 + 0.5) * gScene.screenSize.y);

    GBufferOutput output;
    output.albedo = float4(baseColorSample.rgb * gObject.baseColorFactor.rgb, gObject.params.y * metalRoughness.b);
    output.normal = float4(normalize(input.normal) * 0.5 + 0.5, gObject.params.x * metalRoughness.g);
    // position.w carries the object index + 1 so the lighting pass can fetch per-object material data
    // (0 marks sky, matching the previous validity flag test). motion.zw carries the object UV so
    // the lighting pass can sample the emissive texture with real texture coordinates.
    output.position = float4(input.worldPosition, float(gObject.drawIndex + 1u));
    output.motion = float4(input.position.xy - previousPixel, input.uv);

    // KHR_materials_specular: store the dielectric F0 so the lighting and reflection passes do not
    // need the material data. Missing textures fall back to white and produce the 0.04 default.
    const GpuObjectData objectData = gObjects[gObject.drawIndex];
    const float specularSample =
        gTextures[NonUniformResourceIndex(objectData.specularTextureIndices.x)].Sample(
            gSamplers[NonUniformResourceIndex(objectData.specularSamplerIndices.x)],
            input.uv).r;
    const float3 specularColorSample =
        gTextures[NonUniformResourceIndex(objectData.specularTextureIndices.y)].Sample(
            gSamplers[NonUniformResourceIndex(objectData.specularSamplerIndices.y)],
            input.uv).rgb;
    const float3 specularColor = objectData.specular.yzw * specularColorSample;
    const float specularFactor = objectData.specular.x * specularSample;
    const float ior = max(objectData.volume.y, 1.0);
    const float dielectricF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    output.specularF0 = float4(min(dielectricF0 * specularColor, 1.0) * specularFactor, 1.0);

    // KHR_materials_transmission + KHR_materials_volume: pack the dielectric volume data for the
    // ray traced specular pass.
    const float transmissionSample =
        gTextures[NonUniformResourceIndex(objectData.volumeTextureIndices.x)].Sample(
            gSamplers[NonUniformResourceIndex(objectData.volumeSamplerIndices.x)],
            input.uv).r;
    const float thicknessSample =
        gTextures[NonUniformResourceIndex(objectData.volumeTextureIndices.y)].Sample(
            gSamplers[NonUniformResourceIndex(objectData.volumeSamplerIndices.y)],
            input.uv).g;
    const float transmission = saturate(objectData.volume.x * transmissionSample);
    const float thickness = max(objectData.volume.z * thicknessSample, 0.0);
    // transmission target: (transmission, ior, thickness, attenuationDistance)
    // attenuation target: (attenuationColor.rgb, dispersion)
    output.transmission = float4(transmission, ior, thickness, objectData.attenuation.a);
    output.attenuation = float4(objectData.attenuation.rgb, max(objectData.volume.w, 0.0));

    // KHR_materials_sheen: rgb = sheen color, a = sheen roughness.
    const float3 sheenColorSample =
        gTextures[NonUniformResourceIndex(objectData.sheenTextureIndices.x)].Sample(
            gSamplers[NonUniformResourceIndex(objectData.sheenSamplerIndices.x)],
            input.uv).rgb;
    const float sheenRoughnessSample =
        gTextures[NonUniformResourceIndex(objectData.sheenTextureIndices.y)].Sample(
            gSamplers[NonUniformResourceIndex(objectData.sheenSamplerIndices.y)],
            input.uv).a;
    output.sheen = float4(
        objectData.sheen.rgb * sheenColorSample,
        saturate(objectData.sheen.a * sheenRoughnessSample));
    return output;
}
