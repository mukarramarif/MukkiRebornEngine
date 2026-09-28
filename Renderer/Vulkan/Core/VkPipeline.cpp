#include "Renderer/Vulkan/Core/VkPipeline.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Core/VkShader.h"
#include "Renderer/Vulkan/Resources/DescriptorSet.h"
#include "Utilities/Assert.h"

#include <array>
#include <string>
#include <vector>

namespace mukki::vulkan
{

VkPipeline::VkPipeline(const vk::raii::Device& device, const rhi::PipelineDesc& desc)
{
    vk::PipelineLayoutCreateInfo layoutInfo{};
    const vk::PushConstantRange pushConstantRange = vk::PushConstantRange{}
                                                        .setStageFlags(ToVkShaderStageMask(desc.pushConstantStages))
                                                        .setOffset(0)
                                                        .setSize(desc.pushConstantSize);
    if (desc.pushConstantSize > 0)
    {
        layoutInfo.setPushConstantRanges(pushConstantRange);
    }

    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts{};
    descriptorSetLayouts.reserve(desc.descriptorSetLayouts.size());
    for (const rhi::DescriptorSetLayout* layout : desc.descriptorSetLayouts)
    {
        MRE_ASSERT(layout != nullptr, "PipelineDesc contains a null descriptor set layout");
        descriptorSetLayouts.push_back(static_cast<const VkDescriptorSetLayout*>(layout)->GetHandle());
    }
    if (!descriptorSetLayouts.empty())
    {
        layoutInfo.setSetLayouts(descriptorSetLayouts);
    }

    m_layout = vk::raii::PipelineLayout{device, layoutInfo};

    if (desc.computeShader != nullptr)
    {
        const auto* computeShader = static_cast<const VkShader*>(desc.computeShader);
        MRE_ASSERT(computeShader->GetStage() == rhi::ShaderStage::Compute, "The compute shader has the wrong stage");

        const std::string computeEntryPoint{desc.computeEntryPoint};
        const vk::PipelineShaderStageCreateInfo computeStage = vk::PipelineShaderStageCreateInfo{}
                                                                   .setStage(vk::ShaderStageFlagBits::eCompute)
                                                                   .setModule(computeShader->GetHandle())
                                                                   .setPName(computeEntryPoint.c_str());
        const vk::ComputePipelineCreateInfo pipelineInfo =
            vk::ComputePipelineCreateInfo{}.setStage(computeStage).setLayout(*m_layout);

        std::vector<vk::raii::Pipeline> pipelines = device.createComputePipelines(nullptr, pipelineInfo);
        MRE_ASSERT(pipelines.size() == 1, "Expected exactly one compute pipeline");
        m_pipeline = std::move(pipelines.front());
        m_isCompute = true;
        return;
    }

    MRE_ASSERT(desc.vertexShader != nullptr, "PipelineDesc requires a vertex shader");
    MRE_ASSERT(desc.fragmentShader != nullptr, "PipelineDesc requires a fragment shader");
    MRE_ASSERT(
        !desc.colorFormats.empty() || desc.depthFormat != rhi::Format::Undefined,
        "PipelineDesc requires at least one color format or a depth format");

    const auto* vertexShader = static_cast<const VkShader*>(desc.vertexShader);
    const auto* fragmentShader = static_cast<const VkShader*>(desc.fragmentShader);
    MRE_ASSERT(vertexShader->GetStage() == rhi::ShaderStage::Vertex, "The vertex shader has the wrong stage");
    MRE_ASSERT(fragmentShader->GetStage() == rhi::ShaderStage::Fragment, "The fragment shader has the wrong stage");

    const std::string vertexEntryPoint{desc.vertexEntryPoint};
    const std::string fragmentEntryPoint{desc.fragmentEntryPoint};

    const std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages{
        vk::PipelineShaderStageCreateInfo{}
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(vertexShader->GetHandle())
            .setPName(vertexEntryPoint.c_str()),
        vk::PipelineShaderStageCreateInfo{}
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(fragmentShader->GetHandle())
            .setPName(fragmentEntryPoint.c_str()),
    };

    const bool hasVertexInput = desc.vertexLayout.GetStride() > 0;

    const std::vector<vk::VertexInputBindingDescription> bindingDescriptions = hasVertexInput
        ? std::vector<vk::VertexInputBindingDescription>{vk::VertexInputBindingDescription{}
                                                             .setBinding(0)
                                                             .setStride(desc.vertexLayout.GetStride())
                                                             .setInputRate(vk::VertexInputRate::eVertex)}
        : std::vector<vk::VertexInputBindingDescription>{};

    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions{};
    attributeDescriptions.reserve(desc.vertexLayout.GetAttributes().size());
    for (const rhi::VertexAttribute& attribute : desc.vertexLayout.GetAttributes())
    {
        attributeDescriptions.push_back(vk::VertexInputAttributeDescription{}
                                            .setLocation(attribute.location)
                                            .setBinding(0)
                                            .setFormat(ToVkFormat(attribute.format))
                                            .setOffset(attribute.offset));
    }

    const vk::PipelineVertexInputStateCreateInfo vertexInputState =
        vk::PipelineVertexInputStateCreateInfo{}
            .setVertexBindingDescriptions(bindingDescriptions)
            .setVertexAttributeDescriptions(attributeDescriptions);

    const vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo{}
                                                                            .setTopology(vk::PrimitiveTopology::eTriangleList)
                                                                            .setPrimitiveRestartEnable(false);

    const std::array<vk::DynamicState, 3> dynamicStates{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
        vk::DynamicState::eDepthBias};
    const vk::PipelineDynamicStateCreateInfo dynamicState = vk::PipelineDynamicStateCreateInfo{}.setDynamicStates(
        dynamicStates);

    const vk::PipelineViewportStateCreateInfo viewportState = vk::PipelineViewportStateCreateInfo{}
                                                                  .setViewportCount(1)
                                                                  .setScissorCount(1);

    const vk::PipelineRasterizationStateCreateInfo rasterizationState =
        vk::PipelineRasterizationStateCreateInfo{}
            .setDepthClampEnable(false)
            .setRasterizerDiscardEnable(false)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setCullMode(ToVkCullMode(desc.cullMode))
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setDepthBiasEnable(desc.depthBiasEnable)
            .setDepthBiasConstantFactor(desc.depthBiasConstant)
            .setDepthBiasSlopeFactor(desc.depthBiasSlope)
            .setDepthBiasClamp(desc.depthBiasClamp)
            .setLineWidth(1.0f);

    const vk::PipelineMultisampleStateCreateInfo multisampleState = vk::PipelineMultisampleStateCreateInfo{}
                                                                        .setRasterizationSamples(
                                                                            vk::SampleCountFlagBits::e1)
                                                                        .setSampleShadingEnable(false);

    const bool hasDepthAttachment = desc.depthFormat != rhi::Format::Undefined;
    const vk::PipelineDepthStencilStateCreateInfo depthStencilState = vk::PipelineDepthStencilStateCreateInfo{}
                                                                          .setDepthTestEnable(
                                                                              hasDepthAttachment && desc.depthTest)
                                                                          .setDepthWriteEnable(
                                                                              hasDepthAttachment && desc.depthWrite)
                                                                          .setDepthCompareOp(vk::CompareOp::eLess)
                                                                          .setDepthBoundsTestEnable(false)
                                                                          .setStencilTestEnable(false);

    const vk::PipelineColorBlendAttachmentState colorBlendAttachment = vk::PipelineColorBlendAttachmentState{}
                                                                           .setBlendEnable(desc.blend)
                                                                           .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
                                                                           .setDstColorBlendFactor(
                                                                               vk::BlendFactor::eOneMinusSrcAlpha)
                                                                           .setColorBlendOp(vk::BlendOp::eAdd)
                                                                           .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
                                                                           .setDstAlphaBlendFactor(
                                                                               vk::BlendFactor::eOneMinusSrcAlpha)
                                                                           .setAlphaBlendOp(vk::BlendOp::eAdd)
                                                                           .setColorWriteMask(
                                                                               vk::ColorComponentFlagBits::eR |
                                                                               vk::ColorComponentFlagBits::eG |
                                                                               vk::ColorComponentFlagBits::eB |
                                                                               vk::ColorComponentFlagBits::eA);

    const std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(
        desc.colorFormats.size(),
        colorBlendAttachment);

    const vk::PipelineColorBlendStateCreateInfo colorBlendState = vk::PipelineColorBlendStateCreateInfo{}
                                                                      .setLogicOpEnable(false)
                                                                      .setAttachments(colorBlendAttachments);

    std::vector<vk::Format> colorFormats{};
    colorFormats.reserve(desc.colorFormats.size());
    for (const rhi::Format format : desc.colorFormats)
    {
        colorFormats.push_back(ToVkFormat(format));
    }

    const vk::PipelineRenderingCreateInfo renderingInfo = vk::PipelineRenderingCreateInfo{}
                                                              .setColorAttachmentFormats(colorFormats)
                                                              .setDepthAttachmentFormat(ToVkFormat(desc.depthFormat));

    const vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo{}
                                                            .setPNext(&renderingInfo)
                                                            .setStages(shaderStages)
                                                            .setPVertexInputState(&vertexInputState)
                                                            .setPInputAssemblyState(&inputAssemblyState)
                                                            .setPViewportState(&viewportState)
                                                            .setPRasterizationState(&rasterizationState)
                                                            .setPMultisampleState(&multisampleState)
                                                            .setPDepthStencilState(&depthStencilState)
                                                            .setPColorBlendState(&colorBlendState)
                                                            .setPDynamicState(&dynamicState)
                                                            .setLayout(*m_layout)
                                                            .setRenderPass(nullptr);

    std::vector<vk::raii::Pipeline> pipelines = device.createGraphicsPipelines(nullptr, pipelineInfo);
    MRE_ASSERT(pipelines.size() == 1, "Expected exactly one graphics pipeline");
    m_pipeline = std::move(pipelines.front());
}

} // namespace mukki::vulkan
