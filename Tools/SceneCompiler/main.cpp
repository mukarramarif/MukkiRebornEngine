#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>

#include "Scene/SceneBinary.h"
#include "Scene/SceneJson.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

namespace
{

void PrintUsage()
{
    std::fprintf(
        stderr,
        "usage: mukki_scene_compiler <scene.json> [scene.mres] [--asset-root <directory>]\n");
}

} // namespace

int main(int argc, char** argv)
{
    std::filesystem::path input{};
    std::filesystem::path output{};
    std::filesystem::path assetRoot{};

    for (int index = 1; index < argc; ++index)
    {
        if (std::strcmp(argv[index], "--asset-root") == 0 && index + 1 < argc)
        {
            assetRoot = argv[++index];
        }
        else if (input.empty())
        {
            input = argv[index];
        }
        else if (output.empty())
        {
            output = argv[index];
        }
    }

    if (input.empty())
    {
        PrintUsage();
        return EXIT_FAILURE;
    }

    if (output.empty())
    {
        output = input;
        output.replace_extension(".mres");
    }

    mukki::Log::Init(mukki::LogConfig{});

    int exitCode = EXIT_SUCCESS;
    try
    {
        mukki::SceneDescription scene{};
        std::string error{};
        if (!mukki::SceneJson::Load(input, scene, error))
        {
            MRE_ERROR("{}", error);
            mukki::Log::Shutdown();
            return EXIT_FAILURE;
        }

        if (!assetRoot.empty())
        {
            std::error_code pathError{};
            const std::filesystem::path canonical = std::filesystem::weakly_canonical(assetRoot, pathError);
            scene.assetRoot = pathError ? assetRoot : canonical;
        }

        if (!mukki::SceneBinary::Write(output, scene, error))
        {
            MRE_ERROR("{}", error);
            mukki::Log::Shutdown();
            return EXIT_FAILURE;
        }

        MRE_INFO(
            "Compiled '{}' -> '{}' ({} objects, {} lights)",
            mukki::PathToUtf8(input),
            mukki::PathToUtf8(output),
            scene.objects.size(),
            scene.lights.size());
    }
    catch (const std::exception& exception)
    {
        MRE_CRITICAL("Unhandled exception: {}", exception.what());
        exitCode = EXIT_FAILURE;
    }

    mukki::Log::Shutdown();
    return exitCode;
}
