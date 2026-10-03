// Shared structs and constants for the GI / ray tracing shaders. Include this before declaring the
// bindings and include gi_shading.slang (functions) after the bindings.

static const float kPi = 3.14159265359;
static const float kMaxGiRadiance = 20.0;

struct GiUniform
{
    float4x4 inverseViewProjection;
    float4x4 previousViewProjection;
    float4x4 viewProjection;
    float4 screenSize;
    float4 halfSize;
    float4 cameraPosition;
    float4 skyColor;              // rgb: color, w: intensity
    float4 sunDirectionIntensity; // xyz: direction toward the light, w: intensity
    float4 sunColor;
    float4 params;  // x: maxDistance, y: frameCounter, z: specularMinRoughness, w: feedbackIntensity
    float4 params2; // x: temporalAlphaMin, y: phiDepth, z: phiNormal, w: phiLuminance
    float4 params3; // x: phiHitDistance, y: intensity, z: sunShadowRay, w: resetHistory
    float4 params4; // x: sunShadowSamples, y: sunAngularRadius (radians), z: traceSamples, w: rtSunMask
    float4 params5; // x: probeRays, y: probeHysteresis, z: probeCount, w: reflectionResetHistory
    float4 params6; // x: reflectionMaxDistance, y: reflectionMaxRoughness, z: reflectionSamples, w: reflectionIntensity
    float4 params7; // x: glassRefractionBounces, y: dispersionEnabled, z: volumeAbsorptionScale, w: unused
};

struct GpuGeometryData
{
    float4 baseColorFactor;
    uint4 bufferIndices;  // x: vertex buffer, y: index buffer, z: vertex stride
    uint4 textureIndices; // x: base color, y: metallic-roughness
    uint4 samplerIndices; // x: base color sampler, y: metallic-roughness sampler
    float4 params;        // x: roughness, y: metalness, z: alpha blend, w: alpha cutoff
    float4 volume;        // x: transmission, y: ior, z: thickness, w: dispersion
    float4 attenuation;   // rgb: attenuation color, a: attenuation distance (0 = none)
    uint4 volumeTextureIndices; // x: transmission texture, y: thickness texture
    uint4 volumeSamplerIndices; // x: transmission sampler, y: thickness sampler
    float4 sheen;               // rgb: sheenColorFactor, a: sheenRoughnessFactor
    uint4 sheenTextureIndices;  // x: sheenColorTexture, y: sheenRoughnessTexture
    uint4 sheenSamplerIndices;
};

struct GpuGiProbe
{
    float4 position; // xyz: world position
    float4 sh0;      // L0 rgb
    float4 shX;
    float4 shY;
    float4 shZ;
};
