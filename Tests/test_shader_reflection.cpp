#include "Renderer/RHI/Resources.h"

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace
{

class FakeShader final : public mukki::rhi::Shader
{
public:
    FakeShader(mukki::rhi::ShaderStage stage, mukki::rhi::ShaderReflection reflection)
        : m_reflection(std::move(reflection))
    {
        m_reflection.stage = stage;
    }

    [[nodiscard]] mukki::rhi::ShaderStage GetStage() const override { return m_reflection.stage; }
    [[nodiscard]] const mukki::rhi::ShaderReflection& GetReflection() const override
    {
        return m_reflection;
    }

private:
    mukki::rhi::ShaderReflection m_reflection{};
};

TEST(ShaderReflection, MergesBindingsOfOneSetAcrossStages)
{
    mukki::rhi::ShaderReflection vertexReflection{};
    vertexReflection.bindings = {
        {0, 1, mukki::rhi::DescriptorType::UniformBuffer, 1, mukki::rhi::ShaderStageMask::Vertex},
        {0, 0, mukki::rhi::DescriptorType::UniformBuffer, 1, mukki::rhi::ShaderStageMask::Vertex},
        {1, 0, mukki::rhi::DescriptorType::SampledImage, 64, mukki::rhi::ShaderStageMask::Vertex}};

    mukki::rhi::ShaderReflection fragmentReflection{};
    fragmentReflection.bindings = {
        {0, 1, mukki::rhi::DescriptorType::UniformBuffer, 1, mukki::rhi::ShaderStageMask::Fragment},
        {0, 2, mukki::rhi::DescriptorType::CombinedImageSampler, 1, mukki::rhi::ShaderStageMask::Fragment}};

    const FakeShader vertexShader{mukki::rhi::ShaderStage::Vertex, vertexReflection};
    const FakeShader fragmentShader{mukki::rhi::ShaderStage::Fragment, fragmentReflection};
    const std::array<const mukki::rhi::Shader*, 2> shaders{&vertexShader, &fragmentShader};

    const std::vector<mukki::rhi::DescriptorBinding> merged =
        mukki::rhi::MergeShaderReflectionBindings(shaders, 0);

    ASSERT_EQ(merged.size(), 3u);
    EXPECT_EQ(merged[0].binding, 0u);
    EXPECT_EQ(merged[1].binding, 1u);
    EXPECT_EQ(merged[2].binding, 2u);
    EXPECT_EQ(merged[0].type, mukki::rhi::DescriptorType::UniformBuffer);
    EXPECT_EQ(
        static_cast<std::uint8_t>(merged[1].stages),
        static_cast<std::uint8_t>(mukki::rhi::ShaderStageMask::VertexFragment));
    EXPECT_EQ(
        merged[2].stages,
        mukki::rhi::ShaderStageMask::Fragment);
}

TEST(ShaderReflection, ApplyFillsVertexLayoutAndPushConstants)
{
    mukki::rhi::ShaderReflection vertexReflection{};
    // Deliberately out of order to check that attributes get sorted by location.
    vertexReflection.vertexInputs = {
        {2, mukki::rhi::Format::R32G32Sfloat, 0},
        {0, mukki::rhi::Format::R32G32B32Sfloat, 0},
        {1, mukki::rhi::Format::R32G32B32A32Sfloat, 0}};
    vertexReflection.pushConstants = {
        {0, 112, mukki::rhi::ShaderStageMask::Vertex},
        {0, 64, mukki::rhi::ShaderStageMask::Vertex}};

    mukki::rhi::ShaderReflection fragmentReflection{};
    fragmentReflection.vertexInputs = {{0, mukki::rhi::Format::R32G32Sfloat, 0}};
    fragmentReflection.pushConstants = {
        {0, 112, mukki::rhi::ShaderStageMask::Fragment}};

    const FakeShader vertexShader{mukki::rhi::ShaderStage::Vertex, vertexReflection};
    const FakeShader fragmentShader{mukki::rhi::ShaderStage::Fragment, fragmentReflection};
    const std::array<const mukki::rhi::Shader*, 2> shaders{&vertexShader, &fragmentShader};

    mukki::rhi::PipelineDesc desc{};
    mukki::rhi::ApplyShaderReflection(desc, shaders);

    ASSERT_EQ(desc.vertexLayout.GetAttributes().size(), 3u);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[0].location, 0u);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[0].format, mukki::rhi::Format::R32G32B32Sfloat);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[0].offset, 0u);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[1].location, 1u);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[1].offset, 12u);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[2].location, 2u);
    EXPECT_EQ(desc.vertexLayout.GetAttributes()[2].offset, 28u);
    EXPECT_EQ(desc.vertexLayout.GetStride(), 36u);

    EXPECT_EQ(desc.pushConstantSize, 112u);
    EXPECT_EQ(
        static_cast<std::uint8_t>(desc.pushConstantStages),
        static_cast<std::uint8_t>(mukki::rhi::ShaderStageMask::VertexFragment));
}

TEST(ShaderReflection, VertexLayoutBindingStrideOverride)
{
    mukki::rhi::VertexLayout layout{};
    layout.AddAttribute(0, mukki::rhi::Format::R32G32B32Sfloat);
    EXPECT_EQ(layout.GetStride(), 12u);

    layout.SetBindingStride(32);
    EXPECT_EQ(layout.GetStride(), 32u);
    EXPECT_EQ(layout.GetAttributes()[0].offset, 0u);
}

} // namespace
