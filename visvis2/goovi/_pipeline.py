import vulkan as vk

from ._core import GPUObject


class BasePipeline(GPUObject):

    def __init__(self, device):
        self._device = device


class ComputePipeline(BasePipeline):
    def __init__(self, device):
        raise NotImplementedError()


class GraphicsPipeline(BasePipeline):

    def __init__(self, device, renderpass, shader_modules):
        self._device = device
        self._handle = None
        self._pipelineLayout = None
        self._createGraphicsPipeline(renderpass, shader_modules)

    def destroy(self):
        # todo: this does not get called?

        device_handle = self._device._handle

        if self._handle:
            vkDestroyPipeline(device_handle, self._handle, None)
        self._handle = None

        if self._pipelineLayout:
            vk.vkDestroyPipelineLayout(device_handle, self._pipelineLayout, None)
        self._pipelineLayout = None

    def _createGraphicsPipeline(self, renderpass, shader_modules):
        # There are two types of Pipelines – ComputePipeline and GraphicsPipeline.
        # ComputePipeline is the simpler one, because all it supports is compute-only programs.
        # For each different set of parameters needed during rendering
        # you must create a new Pipeline. You can then set it as the
        # current active Pipeline in a CommandBuffer by calling the
        # function vkCmdBindPipeline .
        # There is also a helper object called PipelineCache, that can
        # be used to speed up pipeline creation. It is a simple object
        # that you can optionally pass in during Pipeline creation, but
        # that really helps to improve performance via reduced memory
        # usage, and the compilation time of your pipelines. The driver
        # can use it internally to store some intermediate data, so
        # that the creation of similar Pipelines could potentially be
        # faster. You can also save and load the state of a
        # PipelineCache object to a buffer of binary data, to save it
        # on disk and use it the next time your application is executed.
        # We recommend you use them!

        swapChainExtent = self._device._swapChainExtent
        device_handle = self._device._handle

        shader_stages = []

        if shader_modules.get("vertex", None):
            vertShaderStageInfo = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags=0,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=shader_modules["vertex"],
                pName="main",
            )
            shader_stages.append(vertShaderStageInfo)

        if shader_modules.get("fragment", None):
            fragShaderStageInfo = vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                flags=0,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=shader_modules["fragment"],
                pName="main",
            )
            shader_stages.append(fragShaderStageInfo)

        vertexInputInfo = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=0,
            vertexAttributeDescriptionCount=0,
        )

        inputAssembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=True,
        )

        viewport = vk.VkViewport(
            0.0,
            0.0,
            float(swapChainExtent.width),
            float(swapChainExtent.height),
            0.0,
            1.0,
        )
        scissor = vk.VkRect2D([0, 0], swapChainExtent)
        viewportState = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=viewport,
            scissorCount=1,
            pScissors=scissor,
        )

        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=False,
            rasterizerDiscardEnable=False,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=False,
        )

        multisampling = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=False,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
        )

        # note: this all makes little sense, probably
        colorBlendAttachment = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=True,
            colorBlendOp=vk.VK_BLEND_OP_ADD,
            srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
            alphaBlendOp=vk.VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
            dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
            | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT
            | vk.VK_COLOR_COMPONENT_A_BIT,
        )

        colorBlending = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=False,
            logicOp=vk.VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=colorBlendAttachment,
            blendConstants=[1.0, 1.0, 1.0, 1.0],
        )

        pipelineLayoutInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=0,
            pushConstantRangeCount=0,
        )

        self._pipelineLayout = vk.vkCreatePipelineLayout(
            device_handle, pipelineLayoutInfo, None
        )

        pipelineInfo = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(shader_stages),
            pStages=shader_stages,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportState,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=colorBlending,
            layout=self._pipelineLayout,
            renderPass=renderpass,
            subpass=0,
        )

        self._handle = vk.vkCreateGraphicsPipelines(
            device_handle, vk.VK_NULL_HANDLE, 1, pipelineInfo, None
        )

        for shader_module in shader_modules.values():
            vk.vkDestroyShaderModule(device_handle, shader_module, None)