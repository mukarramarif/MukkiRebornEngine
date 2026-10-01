#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace mukki::rhi
{

struct ShaderReflection;

enum class Format : std::uint8_t
{
    Undefined = 0,
    R8G8B8A8Unorm,
    R8G8B8A8Srgb,
    B8G8R8A8Unorm,
    B8G8R8A8Srgb,
    R16G16B16A16Sfloat,
    R32G32Sfloat,
    R32G32B32Sfloat,
    R32G32B32A32Sfloat,
    D32Sfloat,
};

enum class PresentMode : std::uint8_t
{
    Fifo = 0,
    Mailbox,
    Immediate,
};

enum class BufferUsage : std::uint8_t
{
    Vertex = 0,
    Index,
    Uniform,
    Storage,
    Staging,
};

enum class TextureUsage : std::uint8_t
{
    Sampled = 0,
    ColorAttachment,
    DepthAttachment,
};

enum class ShaderStage : std::uint8_t
{
    Vertex = 0,
    Fragment,
    Compute,
};

enum class ShaderStageMask : std::uint8_t
{
    None = 0,
    Vertex = 1,
    Fragment = 2,
    Compute = 4,
    VertexFragment = 3,
};

enum class CullMode : std::uint8_t
{
    None = 0,
    Front,
    Back,
};

enum class Filter : std::uint8_t
{
    Nearest = 0,
    Linear,
};

enum class AddressMode : std::uint8_t
{
    Repeat = 0,
    ClampToEdge,
    MirroredRepeat,
};

enum class DescriptorType : std::uint8_t
{
    UniformBuffer = 0,
    StorageBuffer,
    CombinedImageSampler,
    SampledImage,
    Sampler,
};

inline constexpr std::uint32_t kMaxBindlessTextures = 4096;
inline constexpr std::uint32_t kMaxBindlessSamplers = 16;

struct Extent2D
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ClearColor
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

struct BufferDesc
{
    std::uint64_t size = 0;
    BufferUsage usage = BufferUsage::Vertex;
    bool hostVisible = false;
};

class Buffer
{
public:
    virtual ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] virtual std::uint64_t GetSize() const = 0;
    [[nodiscard]] virtual bool IsHostVisible() const = 0;
    virtual void Write(const void* data, std::uint64_t size, std::uint64_t offset = 0) = 0;

protected:
    Buffer() = default;
};

struct TextureDesc
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Format format = Format::Undefined;
    TextureUsage usage = TextureUsage::Sampled;
    const void* initialData = nullptr;
    std::uint64_t initialDataSize = 0;
    std::uint32_t arrayLayers = 1;
};

class Texture
{
public:
    virtual ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    [[nodiscard]] virtual std::uint32_t GetWidth() const = 0;
    [[nodiscard]] virtual std::uint32_t GetHeight() const = 0;
    [[nodiscard]] virtual Format GetFormat() const = 0;
    [[nodiscard]] virtual TextureUsage GetUsage() const = 0;
    [[nodiscard]] virtual std::uint32_t GetArrayLayers() const = 0;
    [[nodiscard]] virtual std::uint32_t GetBindlessIndex() const = 0;

protected:
    Texture() = default;
};

enum class CompareOp : std::uint8_t
{
    Never = 0,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

struct SamplerDesc
{
    Filter minFilter = Filter::Linear;
    Filter magFilter = Filter::Linear;
    AddressMode addressModeU = AddressMode::ClampToEdge;
    AddressMode addressModeV = AddressMode::ClampToEdge;
    bool compareEnable = false;
    CompareOp compareOp = CompareOp::LessEqual;
};

class Sampler
{
public:
    virtual ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

protected:
    Sampler() = default;
};

class Shader
{
public:
    virtual ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    [[nodiscard]] virtual ShaderStage GetStage() const = 0;

    // Interface metadata parsed from the compiled shader, used to generate descriptor set layouts
    // and pipeline state automatically.
    [[nodiscard]] virtual const ShaderReflection& GetReflection() const = 0;

protected:
    Shader() = default;
};

struct VertexAttribute
{
    std::uint32_t location = 0;
    Format format = Format::Undefined;
    std::uint32_t offset = 0;
};

class VertexLayout
{
public:
    void AddAttribute(std::uint32_t location, Format format);

    // Overrides the vertex binding stride for pipelines that only consume a prefix of the vertex
    // format (for example a depth-only shadow pipeline). Attribute offsets keep using the stride
    // derived from the added attributes.
    void SetBindingStride(std::uint32_t stride) { m_bindingStride = stride; }

    [[nodiscard]] const std::vector<VertexAttribute>& GetAttributes() const { return m_attributes; }
    [[nodiscard]] std::uint32_t GetStride() const
    {
        return m_bindingStride != 0 ? m_bindingStride : m_stride;
    }

private:
    std::vector<VertexAttribute> m_attributes{};
    std::uint32_t m_stride = 0;
    std::uint32_t m_bindingStride = 0;
};

struct ShaderReflectionBinding
{
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    DescriptorType type = DescriptorType::UniformBuffer;
    std::uint32_t arrayCount = 1;
    ShaderStageMask stages = ShaderStageMask::None;
};

struct ShaderReflectionPushConstant
{
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
    ShaderStageMask stages = ShaderStageMask::None;
};

struct ShaderReflection
{
    ShaderStage stage = ShaderStage::Vertex;
    std::vector<ShaderReflectionBinding> bindings{};
    std::vector<ShaderReflectionPushConstant> pushConstants{};
    std::vector<VertexAttribute> vertexInputs{};
};

struct DescriptorBinding
{
    std::uint32_t binding = 0;
    DescriptorType type = DescriptorType::UniformBuffer;
    ShaderStageMask stages = ShaderStageMask::VertexFragment;
    std::uint32_t arrayCount = 1;
    bool updateAfterBind = false;
};

// Merges the reflected bindings of the given shader stages for one descriptor set. Bindings shared
// by several stages get their stage masks combined.
[[nodiscard]] std::vector<DescriptorBinding> MergeShaderReflectionBindings(
    std::span<const Shader* const> shaders,
    std::uint32_t setIndex);

class DescriptorSetLayout
{
public:
    virtual ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

protected:
    DescriptorSetLayout() = default;
};

class DescriptorSet
{
public:
    virtual ~DescriptorSet();

    DescriptorSet(const DescriptorSet&) = delete;
    DescriptorSet& operator=(const DescriptorSet&) = delete;

    virtual void BindBuffer(
        std::uint32_t binding,
        const Buffer& buffer,
        std::uint64_t offset = 0,
        std::uint64_t range = 0) = 0;
    virtual void BindTexture(std::uint32_t binding, const Texture& texture, const Sampler& sampler) = 0;
    virtual void BindTexture(std::uint32_t binding, const Texture& texture, std::uint32_t arrayIndex) = 0;
    virtual void BindSampler(std::uint32_t binding, const Sampler& sampler) = 0;
    virtual void BindSampler(std::uint32_t binding, const Sampler& sampler, std::uint32_t arrayIndex) = 0;

protected:
    DescriptorSet() = default;
};

struct PipelineDesc
{
    const Shader* vertexShader = nullptr;
    const Shader* fragmentShader = nullptr;
    const Shader* computeShader = nullptr;
    std::string_view vertexEntryPoint = "vertexMain";
    std::string_view fragmentEntryPoint = "fragmentMain";
    std::string_view computeEntryPoint = "computeMain";
    VertexLayout vertexLayout{};
    bool depthTest = false;
    bool depthWrite = false;
    bool depthBiasEnable = false;
    float depthBiasConstant = 0.0f;
    float depthBiasSlope = 0.0f;
    float depthBiasClamp = 0.0f;
    bool blend = false;
    CullMode cullMode = CullMode::Back;
    std::uint32_t pushConstantSize = 0;
    ShaderStageMask pushConstantStages = ShaderStageMask::Vertex;
    std::vector<const DescriptorSetLayout*> descriptorSetLayouts{};
    std::vector<Format> colorFormats{};
    Format depthFormat = Format::Undefined;
};

// Fills the reflection-driven parts of a pipeline description from the given shader stages: the
// vertex input layout and the push constant range. Descriptor set layouts are not touched; create
// them with Device::CreateDescriptorSetLayout and MergeShaderReflectionBindings.
void ApplyShaderReflection(PipelineDesc& desc, std::span<const Shader* const> shaders);

class Pipeline
{
public:
    virtual ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    [[nodiscard]] virtual bool IsCompute() const = 0;
    [[nodiscard]] virtual std::uint32_t GetPushConstantSize() const = 0;

protected:
    Pipeline() = default;
};

[[nodiscard]] std::uint32_t GetFormatSize(Format format);

} // namespace mukki::rhi
