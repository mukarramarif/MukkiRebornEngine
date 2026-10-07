#include "Assets/GltfLoader.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"
#include "Utilities/Profiling.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <glm/common.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace mukki
{
namespace
{

struct AccessorView
{
    const std::uint8_t* data = nullptr;
    std::size_t stride = 0;
    std::size_t count = 0;
    int componentType = 0;
    int componentCount = 0;
    bool normalized = false;
};

bool GetAccessorView(const tinygltf::Model& gltf, int accessorIndex, AccessorView& out)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(gltf.accessors.size()))
    {
        return false;
    }

    const tinygltf::Accessor& accessor = gltf.accessors[accessorIndex];
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(gltf.bufferViews.size()))
    {
        return false;
    }

    const tinygltf::BufferView& view = gltf.bufferViews[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(gltf.buffers.size()))
    {
        return false;
    }

    int componentCount = 0;
    if (accessor.type == TINYGLTF_TYPE_SCALAR)
    {
        componentCount = 1;
    }
    else if (accessor.type == TINYGLTF_TYPE_VEC2)
    {
        componentCount = 2;
    }
    else if (accessor.type == TINYGLTF_TYPE_VEC3)
    {
        componentCount = 3;
    }
    else if (accessor.type == TINYGLTF_TYPE_VEC4)
    {
        componentCount = 4;
    }
    else
    {
        return false;
    }

    std::size_t componentSize = 0;
    switch (accessor.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        componentSize = 4;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        componentSize = 2;
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        componentSize = 1;
        break;
    default:
        return false;
    }

    const tinygltf::Buffer& buffer = gltf.buffers[view.buffer];
    out.data = buffer.data.data() + view.byteOffset + accessor.byteOffset;
    out.stride = view.byteStride != 0 ? view.byteStride : componentSize * static_cast<std::size_t>(componentCount);
    out.count = accessor.count;
    out.componentType = accessor.componentType;
    out.componentCount = componentCount;
    out.normalized = accessor.normalized;
    return true;
}

float ReadComponent(const std::uint8_t* element, int componentIndex, int componentType, bool normalized)
{
    switch (componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
    {
        float value = 0.0f;
        std::memcpy(&value, element + componentIndex * 4, sizeof(float));
        return value;
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    {
        std::uint32_t value = 0;
        std::memcpy(&value, element + componentIndex * 4, sizeof(std::uint32_t));
        return static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    {
        std::uint16_t value = 0;
        std::memcpy(&value, element + componentIndex * 2, sizeof(std::uint16_t));
        return normalized ? static_cast<float>(value) / 65535.0f : static_cast<float>(value);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    {
        const std::uint8_t value = element[componentIndex];
        return normalized ? static_cast<float>(value) / 255.0f : static_cast<float>(value);
    }
    default:
        return 0.0f;
    }
}

bool ReadFloats(
    const tinygltf::Model& gltf,
    int accessorIndex,
    int expectedComponents,
    std::vector<float>& out)
{
    AccessorView view{};
    if (!GetAccessorView(gltf, accessorIndex, view) || view.componentCount != expectedComponents)
    {
        return false;
    }

    out.resize(view.count * static_cast<std::size_t>(expectedComponents));
    for (std::size_t element = 0; element < view.count; ++element)
    {
        const std::uint8_t* source = view.data + element * view.stride;
        for (int component = 0; component < expectedComponents; ++component)
        {
            out[element * static_cast<std::size_t>(expectedComponents) + component] =
                ReadComponent(source, component, view.componentType, view.normalized);
        }
    }

    return true;
}

bool ReadIndices(const tinygltf::Model& gltf, int accessorIndex, std::vector<std::uint32_t>& out)
{
    AccessorView view{};
    if (!GetAccessorView(gltf, accessorIndex, view) || view.componentCount != 1)
    {
        return false;
    }

    out.resize(view.count);
    for (std::size_t element = 0; element < view.count; ++element)
    {
        const std::uint8_t* source = view.data + element * view.stride;
        switch (view.componentType)
        {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        {
            std::uint32_t value = 0;
            std::memcpy(&value, source, sizeof(std::uint32_t));
            out[element] = value;
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        {
            std::uint16_t value = 0;
            std::memcpy(&value, source, sizeof(std::uint16_t));
            out[element] = value;
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            out[element] = source[0];
            break;
        default:
            return false;
        }
    }

    return true;
}

void ConvertImageToRgba(const tinygltf::Image& source, GltfImage& out)
{
    out.width = static_cast<std::uint32_t>(std::max(source.width, 0));
    out.height = static_cast<std::uint32_t>(std::max(source.height, 0));

    const std::size_t pixelCount = static_cast<std::size_t>(out.width) * out.height;
    out.pixels.assign(pixelCount * 4, 255);

    const int components = source.component;
    const int bits = source.bits;
    if (components <= 0 || pixelCount == 0)
    {
        return;
    }

    const std::size_t bytesPerComponent = bits == 16 ? 2 : 1;
    const std::size_t requiredBytes = pixelCount * static_cast<std::size_t>(components) * bytesPerComponent;
    if ((bits != 8 && bits != 16) || source.image.size() < requiredBytes)
    {
        MRE_WARN("Unsupported glTF image format ({} bits, {} components)", bits, components);
        return;
    }

    const auto sampleChannel = [&source, components, bits, bytesPerComponent](
                                   std::size_t pixel,
                                   int component) -> std::uint8_t {
        const std::size_t offset = (pixel * static_cast<std::size_t>(components) +
                                    static_cast<std::size_t>(component)) *
                                   bytesPerComponent;
        if (bits == 16)
        {
            std::uint16_t value = 0;
            std::memcpy(&value, source.image.data() + offset, sizeof(std::uint16_t));
            return static_cast<std::uint8_t>((static_cast<std::uint32_t>(value) * 255u + 32767u) / 65535u);
        }
        return source.image[offset];
    };

    for (std::size_t pixel = 0; pixel < pixelCount; ++pixel)
    {
        std::uint8_t* dst = out.pixels.data() + pixel * 4;

        if (components >= 3)
        {
            dst[0] = sampleChannel(pixel, 0);
            dst[1] = sampleChannel(pixel, 1);
            dst[2] = sampleChannel(pixel, 2);
            dst[3] = components >= 4 ? sampleChannel(pixel, 3) : 255;
        }
        else if (components == 1)
        {
            const std::uint8_t gray = sampleChannel(pixel, 0);
            dst[0] = gray;
            dst[1] = gray;
            dst[2] = gray;
            dst[3] = 255;
        }
        else
        {
            const std::uint8_t gray = sampleChannel(pixel, 0);
            dst[0] = gray;
            dst[1] = gray;
            dst[2] = gray;
            dst[3] = sampleChannel(pixel, 1);
        }
    }
}

GltfWrapMode ConvertWrapMode(int wrap)
{
    switch (wrap)
    {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
        return GltfWrapMode::ClampToEdge;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
        return GltfWrapMode::MirroredRepeat;
    case TINYGLTF_TEXTURE_WRAP_REPEAT:
    default:
        return GltfWrapMode::Repeat;
    }
}

glm::mat4 GetNodeLocalTransform(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        return glm::make_mat4(node.matrix.data());
    }

    glm::vec3 translation{0.0f};
    if (node.translation.size() == 3)
    {
        translation = glm::vec3{
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2])};
    }

    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    if (node.rotation.size() == 4)
    {
        rotation = glm::quat{
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2])};
    }

    glm::vec3 scale{1.0f};
    if (node.scale.size() == 3)
    {
        scale = glm::vec3{
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2])};
    }

    glm::mat4 transform = glm::translate(glm::mat4{1.0f}, translation);
    transform *= glm::mat4_cast(rotation);
    transform = glm::scale(transform, scale);
    return transform;
}

void ExpandBounds(const GltfPrimitive& primitive, const glm::mat4& transform, glm::vec3& minBounds, glm::vec3& maxBounds)
{
    for (const GltfVertex& vertex : primitive.vertices)
    {
        const glm::vec4 worldPosition = transform * glm::vec4{vertex.position, 1.0f};
        minBounds = glm::min(minBounds, glm::vec3{worldPosition});
        maxBounds = glm::max(maxBounds, glm::vec3{worldPosition});
    }
}

void TraverseNode(
    const tinygltf::Model& gltf,
    int nodeIndex,
    const glm::mat4& parentTransform,
    const std::vector<std::vector<std::uint32_t>>& meshPrimitives,
    GltfModel& out)
{
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(gltf.nodes.size()))
    {
        return;
    }

    const tinygltf::Node& node = gltf.nodes[nodeIndex];
    const glm::mat4 worldTransform = parentTransform * GetNodeLocalTransform(node);

    if (node.mesh >= 0 && node.mesh < static_cast<int>(meshPrimitives.size()))
    {
        out.nodes.push_back(GltfNode{worldTransform, node.mesh});

        for (const std::uint32_t primitiveIndex : meshPrimitives[static_cast<std::size_t>(node.mesh)])
        {
            ExpandBounds(out.primitives[primitiveIndex], worldTransform, out.minBounds, out.maxBounds);
        }
    }

    for (const int child : node.children)
    {
        TraverseNode(gltf, child, worldTransform, meshPrimitives, out);
    }
}

} // namespace

bool LoadGltfModel(const std::filesystem::path& path, GltfModel& model)
{
    MRE_PROFILE_ZONE_N("LoadGltfModel");

    std::error_code errorCode{};
    if (!std::filesystem::exists(path, errorCode))
    {
        MRE_ERROR("glTF file '{}' does not exist", PathToUtf8(path));
        return false;
    }

    std::ifstream stream{path, std::ios::in | std::ios::binary};
    if (!stream.is_open())
    {
        MRE_ERROR("Failed to open glTF file '{}'", PathToUtf8(path));
        return false;
    }

    const std::vector<char> bytes{
        std::istreambuf_iterator<char>{stream},
        std::istreambuf_iterator<char>{}};
    if (bytes.empty())
    {
        MRE_ERROR("glTF file '{}' is empty", PathToUtf8(path));
        return false;
    }

    tinygltf::TinyGLTF context;
    tinygltf::Model gltf{};
    std::string errors{};
    std::string warnings{};
    const std::string baseDirectory = PathToUtf8(path.parent_path());

    const bool isBinary = path.extension() == ".glb";
    const bool loaded = isBinary
                            ? context.LoadBinaryFromMemory(
                                  &gltf,
                                  &errors,
                                  &warnings,
                                  reinterpret_cast<const unsigned char*>(bytes.data()),
                                  static_cast<unsigned int>(bytes.size()),
                                  baseDirectory)
                            : context.LoadASCIIFromString(
                                  &gltf,
                                  &errors,
                                  &warnings,
                                  bytes.data(),
                                  static_cast<unsigned int>(bytes.size()),
                                  baseDirectory);

    if (!warnings.empty())
    {
        MRE_WARN("glTF loader warnings for '{}': {}", PathToUtf8(path.filename()), warnings);
    }

    if (!loaded)
    {
        MRE_ERROR("Failed to load '{}': {}", PathToUtf8(path), errors);
        return false;
    }

    model = GltfModel{};
    model.minBounds = glm::vec3{std::numeric_limits<float>::max()};
    model.maxBounds = glm::vec3{std::numeric_limits<float>::lowest()};

    for (const tinygltf::Image& image : gltf.images)
    {
        GltfImage converted{};
        ConvertImageToRgba(image, converted);
        model.images.push_back(std::move(converted));
    }

    for (const tinygltf::Texture& texture : gltf.textures)
    {
        GltfTexture converted{};
        converted.imageIndex = texture.source;
        if (texture.sampler >= 0 && texture.sampler < static_cast<int>(gltf.samplers.size()))
        {
            const tinygltf::Sampler& sampler = gltf.samplers[static_cast<std::size_t>(texture.sampler)];
            converted.wrapS = ConvertWrapMode(sampler.wrapS);
            converted.wrapT = ConvertWrapMode(sampler.wrapT);
        }
        model.textures.push_back(converted);
    }

    for (const tinygltf::Material& material : gltf.materials)
    {
        GltfMaterial converted{};
        if (material.pbrMetallicRoughness.baseColorFactor.size() == 4)
        {
            converted.baseColorFactor = glm::vec4{
                static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[0]),
                static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[1]),
                static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[2]),
                static_cast<float>(material.pbrMetallicRoughness.baseColorFactor[3])};
        }

        converted.metallicFactor = static_cast<float>(material.pbrMetallicRoughness.metallicFactor);
        converted.roughnessFactor = static_cast<float>(material.pbrMetallicRoughness.roughnessFactor);
        converted.doubleSided = material.doubleSided;
        converted.alphaMask = material.alphaMode == "MASK";
        converted.alphaBlend = material.alphaMode == "BLEND";
        converted.alphaCutoff = static_cast<float>(material.alphaCutoff);

        const auto extensionValue = [&material](const char* name) -> const tinygltf::Value* {
            const auto found = material.extensions.find(name);
            return found == material.extensions.end() ? nullptr : &found->second;
        };
        const auto readNumber = [](const tinygltf::Value& value, const char* key, float fallback) {
            if (!value.Has(key) || !value.Get(key).IsNumber())
            {
                return fallback;
            }
            return static_cast<float>(value.Get(key).GetNumberAsDouble());
        };
        const auto readColor = [](const tinygltf::Value& value, const char* key, const glm::vec3& fallback) {
            if (!value.Has(key))
            {
                return fallback;
            }
            const tinygltf::Value& color = value.Get(key);
            if (!color.IsArray() || color.ArrayLen() != 3)
            {
                return fallback;
            }
            return glm::vec3{
                static_cast<float>(color.Get(0).GetNumberAsDouble()),
                static_cast<float>(color.Get(1).GetNumberAsDouble()),
                static_cast<float>(color.Get(2).GetNumberAsDouble())};
        };
        const auto readTextureIndex = [](const tinygltf::Value& value, const char* key) -> int {
            if (!value.Has(key))
            {
                return -1;
            }
            const tinygltf::Value& info = value.Get(key);
            if (!info.IsObject() || !info.Has("index") || !info.Get("index").IsNumber())
            {
                return -1;
            }
            return static_cast<int>(info.Get("index").GetNumberAsInt());
        };

        if (const tinygltf::Value* specular = extensionValue("KHR_materials_specular"))
        {
            converted.specularFactor = readNumber(*specular, "specularFactor", converted.specularFactor);
            converted.specularColorFactor =
                readColor(*specular, "specularColorFactor", converted.specularColorFactor);
            converted.specularTexture = readTextureIndex(*specular, "specularTexture");
            converted.specularColorTexture = readTextureIndex(*specular, "specularColorTexture");
        }

        if (const tinygltf::Value* iorExtension = extensionValue("KHR_materials_ior"))
        {
            converted.ior = readNumber(*iorExtension, "ior", converted.ior);
        }

        if (const tinygltf::Value* transmission = extensionValue("KHR_materials_transmission"))
        {
            converted.transmissionFactor =
                readNumber(*transmission, "transmissionFactor", converted.transmissionFactor);
            converted.transmissionTexture = readTextureIndex(*transmission, "transmissionTexture");
        }

        if (const tinygltf::Value* volume = extensionValue("KHR_materials_volume"))
        {
            converted.thicknessFactor = readNumber(*volume, "thicknessFactor", converted.thicknessFactor);
            converted.thicknessTexture = readTextureIndex(*volume, "thicknessTexture");
            converted.attenuationColor = readColor(*volume, "attenuationColor", converted.attenuationColor);
            converted.attenuationDistance =
                readNumber(*volume, "attenuationDistance", converted.attenuationDistance);
        }

        if (const tinygltf::Value* dispersion = extensionValue("KHR_materials_dispersion"))
        {
            converted.dispersion = readNumber(*dispersion, "dispersion", converted.dispersion);
        }

        if (const tinygltf::Value* sheen = extensionValue("KHR_materials_sheen"))
        {
            converted.sheenColorFactor =
                readColor(*sheen, "sheenColorFactor", converted.sheenColorFactor);
            converted.sheenRoughnessFactor =
                readNumber(*sheen, "sheenRoughnessFactor", converted.sheenRoughnessFactor);
            converted.sheenColorTexture = readTextureIndex(*sheen, "sheenColorTexture");
            converted.sheenRoughnessTexture = readTextureIndex(*sheen, "sheenRoughnessTexture");
        }

        converted.emissiveFactor = glm::vec3{
            static_cast<float>(material.emissiveFactor.size() > 0 ? material.emissiveFactor[0] : 0.0),
            static_cast<float>(material.emissiveFactor.size() > 1 ? material.emissiveFactor[1] : 0.0),
            static_cast<float>(material.emissiveFactor.size() > 2 ? material.emissiveFactor[2] : 0.0)};
        if (const tinygltf::Value* emissiveStrength =
                extensionValue("KHR_materials_emissive_strength"))
        {
            const float strength =
                readNumber(*emissiveStrength, "emissiveStrength", 1.0f);
            converted.emissiveFactor *= std::max(strength, 0.0f);
        }
        const int emissiveTextureIndex = material.emissiveTexture.index;
        if (emissiveTextureIndex >= 0 &&
            emissiveTextureIndex < static_cast<int>(gltf.textures.size()))
        {
            converted.emissiveTexture = emissiveTextureIndex;
        }

        const int baseColorTextureIndex = material.pbrMetallicRoughness.baseColorTexture.index;
        if (baseColorTextureIndex >= 0 && baseColorTextureIndex < static_cast<int>(gltf.textures.size()))
        {
            converted.baseColorTexture = baseColorTextureIndex;
        }

        const int metalRoughnessTextureIndex = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
        if (metalRoughnessTextureIndex >= 0 &&
            metalRoughnessTextureIndex < static_cast<int>(gltf.textures.size()))
        {
            converted.metallicRoughnessTexture = metalRoughnessTextureIndex;
        }

        model.materials.push_back(converted);
    }

    std::vector<std::vector<std::uint32_t>>& meshPrimitives = model.meshPrimitives;
    meshPrimitives.resize(gltf.meshes.size());

    for (std::size_t meshIndex = 0; meshIndex < gltf.meshes.size(); ++meshIndex)
    {
        const tinygltf::Mesh& mesh = gltf.meshes[meshIndex];
        for (const tinygltf::Primitive& primitive : mesh.primitives)
        {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
            {
                continue;
            }

            const auto positionAttribute = primitive.attributes.find("POSITION");
            if (positionAttribute == primitive.attributes.end())
            {
                continue;
            }

            std::vector<float> positions{};
            if (!ReadFloats(gltf, positionAttribute->second, 3, positions))
            {
                continue;
            }

            const std::size_t vertexCount = positions.size() / 3;
            std::vector<float> normals{};
            const auto normalAttribute = primitive.attributes.find("NORMAL");
            if (normalAttribute == primitive.attributes.end() ||
                !ReadFloats(gltf, normalAttribute->second, 3, normals) ||
                normals.size() != vertexCount * 3)
            {
                normals.assign(vertexCount * 3, 0.0f);
            }

            std::vector<float> texCoords{};
            const auto uvAttribute = primitive.attributes.find("TEXCOORD_0");
            if (uvAttribute == primitive.attributes.end() ||
                !ReadFloats(gltf, uvAttribute->second, 2, texCoords) ||
                texCoords.size() != vertexCount * 2)
            {
                texCoords.assign(vertexCount * 2, 0.0f);
            }

            GltfPrimitive converted{};
            converted.vertices.resize(vertexCount);
            for (std::size_t vertex = 0; vertex < vertexCount; ++vertex)
            {
                converted.vertices[vertex].position = glm::vec3{
                    positions[vertex * 3 + 0],
                    positions[vertex * 3 + 1],
                    positions[vertex * 3 + 2]};
                converted.vertices[vertex].normal = glm::vec3{
                    normals[vertex * 3 + 0],
                    normals[vertex * 3 + 1],
                    normals[vertex * 3 + 2]};
                converted.vertices[vertex].uv = glm::vec2{
                    texCoords[vertex * 2 + 0],
                    texCoords[vertex * 2 + 1]};
            }

            if (primitive.indices >= 0)
            {
                if (!ReadIndices(gltf, primitive.indices, converted.indices))
                {
                    converted.indices.clear();
                }
            }

            if (converted.indices.empty())
            {
                converted.indices.resize(vertexCount);
                for (std::size_t vertex = 0; vertex < vertexCount; ++vertex)
                {
                    converted.indices[vertex] = static_cast<std::uint32_t>(vertex);
                }
            }

            converted.materialIndex = primitive.material;

            model.vertexCount += converted.vertices.size();
            model.triangleCount += converted.indices.size() / 3;

            meshPrimitives[meshIndex].push_back(static_cast<std::uint32_t>(model.primitives.size()));
            model.primitives.push_back(std::move(converted));
        }
    }

    if (gltf.defaultScene >= 0 && gltf.defaultScene < static_cast<int>(gltf.scenes.size()))
    {
        for (const int nodeIndex : gltf.scenes[static_cast<std::size_t>(gltf.defaultScene)].nodes)
        {
            TraverseNode(gltf, nodeIndex, glm::mat4{1.0f}, meshPrimitives, model);
        }
    }

    if (model.primitives.empty() || model.nodes.empty())
    {
        MRE_ERROR("glTF file '{}' contains no drawable primitives", PathToUtf8(path));
        return false;
    }

    MRE_INFO(
        "Loaded '{}': {} primitives, {} nodes, {} materials, {} images, {} vertices, {} triangles",
        PathToUtf8(path.filename()),
        model.primitives.size(),
        model.nodes.size(),
        model.materials.size(),
        model.images.size(),
        model.vertexCount,
        model.triangleCount);

    return true;
}

} // namespace mukki
