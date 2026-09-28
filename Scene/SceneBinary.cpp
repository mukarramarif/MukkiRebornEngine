#include "Scene/SceneBinary.h"

#include <cstring>
#include <fstream>
#include <utility>

#include "Utilities/PathUtils.h"

namespace mukki
{
namespace
{

constexpr std::uint32_t kMaxStringLength = 4096;
constexpr std::uint32_t kMaxObjectCount = 100000;
constexpr std::uint32_t kMaxLightCount = 65536;

class Writer
{
public:
    void U32(std::uint32_t value)
    {
        for (std::uint32_t byte = 0; byte < 4; ++byte)
        {
            m_data.push_back(static_cast<std::byte>((value >> (byte * 8)) & 0xFFu));
        }
    }

    void F32(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        U32(bits);
    }

    void String(const std::string& value)
    {
        U32(static_cast<std::uint32_t>(value.size()));
        for (const char character : value)
        {
            m_data.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
        }
    }

    void Vec3(const glm::vec3& value)
    {
        F32(value.x);
        F32(value.y);
        F32(value.z);
    }

    [[nodiscard]] std::vector<std::byte> Take() { return std::move(m_data); }

private:
    std::vector<std::byte> m_data{};
};

class Reader
{
public:
    explicit Reader(std::span<const std::byte> data) : m_data(data) {}

    bool U32(std::uint32_t& out)
    {
        if (!m_ok || m_offset + 4 > m_data.size())
        {
            m_ok = false;
            return false;
        }

        out = 0;
        for (std::uint32_t byte = 0; byte < 4; ++byte)
        {
            const std::uint8_t value = std::to_integer<std::uint8_t>(m_data[m_offset + byte]);
            out |= static_cast<std::uint32_t>(value) << (byte * 8);
        }
        m_offset += 4;
        return true;
    }

    bool F32(float& out)
    {
        std::uint32_t bits = 0;
        if (!U32(bits))
        {
            return false;
        }
        std::memcpy(&out, &bits, sizeof(out));
        return true;
    }

    bool String(std::string& out)
    {
        std::uint32_t length = 0;
        if (!U32(length) || length > kMaxStringLength || m_offset + length > m_data.size())
        {
            m_ok = false;
            return false;
        }

        out.assign(
            reinterpret_cast<const char*>(m_data.data() + m_offset),
            static_cast<std::size_t>(length));
        m_offset += length;
        return true;
    }

    bool Vec3(glm::vec3& out)
    {
        return F32(out.x) && F32(out.y) && F32(out.z);
    }

    [[nodiscard]] bool Ok() const { return m_ok; }

private:
    std::span<const std::byte> m_data{};
    std::size_t m_offset = 0;
    bool m_ok = true;
};

} // namespace

std::vector<std::byte> SceneBinary::Encode(const SceneDescription& scene)
{
    Writer writer;
    writer.U32(kMagic);
    writer.U32(kVersion);
    writer.String(scene.name);
    writer.String(PathToUtf8(scene.assetRoot));

    writer.U32(static_cast<std::uint32_t>(scene.objects.size()));
    for (const SceneObject& object : scene.objects)
    {
        writer.String(object.name);
        writer.String(object.modelPath);
        writer.Vec3(object.position);
        writer.Vec3(object.rotation);
        writer.Vec3(object.scale);
    }

    writer.U32(static_cast<std::uint32_t>(scene.lights.size()));
    for (const SceneLight& sceneLight : scene.lights)
    {
        const Light& light = sceneLight.light;
        writer.String(sceneLight.name);
        writer.U32(static_cast<std::uint32_t>(light.type));
        writer.Vec3(light.position);
        writer.Vec3(light.direction);
        writer.Vec3(light.color);
        writer.F32(light.intensity);
        writer.F32(light.range);
        writer.F32(light.innerConeAngle);
        writer.F32(light.outerConeAngle);
    }

    return writer.Take();
}

bool SceneBinary::Decode(std::span<const std::byte> data, SceneDescription& scene, std::string& error)
{
    error.clear();

    Reader reader{data};
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!reader.U32(magic) || !reader.U32(version))
    {
        error = "scene binary is truncated";
        return false;
    }
    if (magic != kMagic)
    {
        error = "scene binary has an invalid magic header";
        return false;
    }
    if (version != kVersion)
    {
        error = "unsupported scene binary version " + std::to_string(version);
        return false;
    }

    SceneDescription result{};

    std::string name{};
    std::string assetRoot{};
    if (!reader.String(name) || !reader.String(assetRoot))
    {
        error = "scene binary is truncated";
        return false;
    }
    result.name = std::move(name);
    result.assetRoot = PathFromUtf8(assetRoot);

    std::uint32_t objectCount = 0;
    if (!reader.U32(objectCount) || objectCount > kMaxObjectCount)
    {
        error = "scene binary has an invalid object count";
        return false;
    }

    result.objects.reserve(objectCount);
    for (std::uint32_t index = 0; index < objectCount; ++index)
    {
        SceneObject object{};
        if (!reader.String(object.name) || !reader.String(object.modelPath) ||
            !reader.Vec3(object.position) || !reader.Vec3(object.rotation) || !reader.Vec3(object.scale))
        {
            error = "scene binary is truncated while reading objects";
            return false;
        }
        result.objects.push_back(std::move(object));
    }

    std::uint32_t lightCount = 0;
    if (!reader.U32(lightCount) || lightCount > kMaxLightCount)
    {
        error = "scene binary has an invalid light count";
        return false;
    }

    result.lights.reserve(lightCount);
    for (std::uint32_t index = 0; index < lightCount; ++index)
    {
        SceneLight sceneLight{};
        std::uint32_t type = 0;
        if (!reader.String(sceneLight.name) || !reader.U32(type) || !reader.Vec3(sceneLight.light.position) ||
            !reader.Vec3(sceneLight.light.direction) || !reader.Vec3(sceneLight.light.color) ||
            !reader.F32(sceneLight.light.intensity) || !reader.F32(sceneLight.light.range) ||
            !reader.F32(sceneLight.light.innerConeAngle) || !reader.F32(sceneLight.light.outerConeAngle))
        {
            error = "scene binary is truncated while reading lights";
            return false;
        }
        if (type > static_cast<std::uint32_t>(LightType::Spot))
        {
            error = "scene binary contains an unknown light type";
            return false;
        }
        sceneLight.light.type = static_cast<LightType>(type);
        result.lights.push_back(std::move(sceneLight));
    }

    scene = std::move(result);
    return true;
}

bool SceneBinary::Write(const std::filesystem::path& path, const SceneDescription& scene, std::string& error)
{
    const std::vector<std::byte> data = Encode(scene);

    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream)
    {
        error = "failed to open scene binary '" + PathToUtf8(path) + "' for writing";
        return false;
    }

    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!stream)
    {
        error = "failed to write scene binary '" + PathToUtf8(path) + "'";
        return false;
    }
    return true;
}

bool SceneBinary::Read(const std::filesystem::path& path, SceneDescription& scene, std::string& error)
{
    std::ifstream stream{path, std::ios::binary | std::ios::ate};
    if (!stream)
    {
        error = "failed to open scene binary '" + PathToUtf8(path) + "'";
        return false;
    }

    const std::streamsize size = stream.tellg();
    if (size <= 0)
    {
        error = "scene binary '" + PathToUtf8(path) + "' is empty";
        return false;
    }

    stream.seekg(0);
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(data.data()), size);
    if (!stream)
    {
        error = "failed to read scene binary '" + PathToUtf8(path) + "'";
        return false;
    }

    return Decode(data, scene, error);
}

} // namespace mukki
