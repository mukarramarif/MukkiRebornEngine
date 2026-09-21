#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace mukki::rhi
{

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
    CombinedImageSampler,
};

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

protected:
    Texture() = default;
};

struct SamplerDesc
{
    Filter minFilter = Filter::Linear;
    Filter magFilter = Filter::Linear;
    AddressMode addressMode = AddressMode::ClampToEdge;
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

    [[nodiscard]] const std::vector<VertexAttribute>& GetAttributes() const { return m_attributes; }
    [[nodiscard]] std::uint32_t GetStride() const { return m_stride; }

private:
    std::vector<VertexAttribute> m_attributes{};
    std::uint32_t m_stride = 0;
};

struct DescriptorBinding
{
    std::uint32_t binding = 0;
    DescriptorType type = DescriptorType::UniformBuffer;
    ShaderStageMask stages = ShaderStageMask::VertexFragment;
};

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

protected:
    DescriptorSet() = default;
};

struct PipelineDesc
{
    const Shader* vertexShader = nullptr;
    const Shader* fragmentShader = nullptr;
    std::string_view vertexEntryPoint = "vertexMain";
    std::string_view fragmentEntryPoint = "fragmentMain";
    VertexLayout vertexLayout{};
    bool depthTest = false;
    bool depthWrite = false;
    bool blend = false;
    CullMode cullMode = CullMode::Back;
    std::uint32_t pushConstantSize = 0;
    ShaderStageMask pushConstantStages = ShaderStageMask::Vertex;
    std::vector<const DescriptorSetLayout*> descriptorSetLayouts{};
    std::vector<Format> colorFormats{};
    Format depthFormat = Format::Undefined;
};

class Pipeline
{
public:
    virtual ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

protected:
    Pipeline() = default;
};

[[nodiscard]] std::uint32_t GetFormatSize(Format format);

} // namespace mukki::rhi
