import os
import time
import asyncio

import vulkan as vk  # todo: should not need this in examples

from visvis2 import goovi, spirv



class Figure:
    """ Wraps stuff ...
    """

    def __init__(self, instance, surface, logicaldevice):
        self._instance = instance
        self._surface = surface
        self._device = logicaldevice

        self._renderPass = None
        self._pipelineLayout = None
        self._graphicsPipeline = None
        self._swapChainFramebuffers = None
        self._commandPool = None
        self._commandBuffers = None
        self._imageAvailableSemaphore = None
        self._renderFinishedSemaphore = None

        self._renderPass = self._createRenderPass()
        self._createGraphicsPipeline()
        self._createFramebuffers()
        self._createCommandPool()
        self._createCommandBuffers()
        self._createSemaphores()

    def destroy(self):
        # todo: this does not get called

        device_handle = self._device._handle

        if self._imageAvailableSemaphore:
            vk.vkDestroySemaphore(device_handle, self._imageAvailableSemaphore, None)
        self._imageAvailableSemaphore = None

        if self._renderFinishedSemaphore:
            vk.vkDestroySemaphore(device_handle, self._renderFinishedSemaphore, None)
        self._renderFinishedSemaphore = None

        if self._commandBuffers:
            pass
        self._commandBuffers = None

        if self._commandPool:
            vk.vkDestroyCommandPool(device_handle, self._commandPool, None)
        self._commandPool = None

        if self._swapChainFramebuffers:
            for i in self._swapChainFramebuffers:
                vk.vkDestroyFramebuffer(device_handle, i, None)
        self._swapChainFramebuffers = None

        if self._graphicsPipeline:
            vkDestroyPipeline(device_handle, self._graphicsPipeline, None)
        self._graphicsPipeline = None

        if self._pipelineLayout:
            vk.vkDestroyPipelineLayout(device_handle, self._pipelineLayout, None)
        self._pipelineLayout = None

        if self._renderPass:
            vk.vkDestroyRenderPass(device_handle, self._renderPass, None)
        self._renderPass = None


    def _createRenderPass(self):

        swapChainImageFormat = self._device._swapChainImageFormat
        device_handle = self._device._handle

        colorAttachment = vk.VkAttachmentDescription(
            format=swapChainImageFormat,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )

        colorAttachmentRef = vk.VkAttachmentReference(
            attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        subPass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=colorAttachmentRef,
        )

        renderPassInfo = vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=1,
            pAttachments=colorAttachment,
            subpassCount=1,
            pSubpasses=subPass,
        )

        return vk.vkCreateRenderPass(device_handle, renderPassInfo, None)

    def _createGraphicsPipeline(self):

        swapChainExtent = self._device._swapChainExtent
        device_handle = self._device._handle
        renderPass = self._renderPass

        path = os.path.dirname(os.path.abspath(__file__))
        vertShaderModule = spirv.get_vert_shader(device_handle)
        fragShaderModule = spirv.get_frag_shader(device_handle)
        # vertShaderModule = self.__createShaderModule(os.path.join(path, 'hello_triangle_vert.spv'))
        # fragShaderModule = self.__createShaderModule(os.path.join(path, 'hello_triangle_frag.spv'))

        vertShaderStageInfo = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            flags=0,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            module=vertShaderModule,
            pName="main",
        )

        fragShaderStageInfo = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            flags=0,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            module=fragShaderModule,
            pName="main",
        )

        shaderStages = [vertShaderStageInfo, fragShaderStageInfo]

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

        colorBlendAttachment = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
            | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT
            | vk.VK_COLOR_COMPONENT_A_BIT,
            blendEnable=False,
        )

        colorBlending = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=False,
            logicOp=vk.VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=colorBlendAttachment,
            blendConstants=[0.0, 0.0, 0.0, 0.0],
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
            stageCount=2,
            pStages=shaderStages,
            pVertexInputState=vertexInputInfo,
            pInputAssemblyState=inputAssembly,
            pViewportState=viewportState,
            pRasterizationState=rasterizer,
            pMultisampleState=multisampling,
            pColorBlendState=colorBlending,
            layout=self._pipelineLayout,
            renderPass=renderPass,
            subpass=0,
        )

        self._graphicsPipeline = vk.vkCreateGraphicsPipelines(
            device_handle, vk.VK_NULL_HANDLE, 1, pipelineInfo, None
        )

        vk.vkDestroyShaderModule(device_handle, vertShaderModule, None)
        vk.vkDestroyShaderModule(device_handle, fragShaderModule, None)

    def _createFramebuffers(self):

        # todo: put in device, or surface, or ... but needs ref to renderpass

        device_handle = self._device._handle
        swapChainImageViews = self._device._swapChainImageViews
        swapChainExtent = self._device._swapChainExtent
        renderPass = self._renderPass

        swapChainFramebuffers = []

        for imageView in swapChainImageViews:
            attachments = [imageView]
            framebufferInfo = vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=renderPass,
                attachmentCount=1,
                pAttachments=attachments,
                width=swapChainExtent.width,
                height=swapChainExtent.height,
                layers=1,
            )
            framebuffer = vk.vkCreateFramebuffer(device_handle, framebufferInfo, None)
            swapChainFramebuffers.append(framebuffer)

        self._swapChainFramebuffers = swapChainFramebuffers

    def _createCommandPool(self):

        device_handle = self._device._handle
        pdevice_ref = self._device._pdevice._ref

        # todo: yuk
        instance_handle = self._instance._handle
        surface_handle = self._surface._handle
        queueFamilyIndices = goovi._device.findQueueFamilies(instance_handle, surface_handle, pdevice_ref)

        poolInfo = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=queueFamilyIndices.graphicsFamily,
        )

        self._commandPool = vk.vkCreateCommandPool(device_handle, poolInfo, None)

    def _createCommandBuffers(self):

        device_handle = self._device._handle
        commandPool = self._commandPool
        swapChainFramebuffers = self._swapChainFramebuffers
        renderPass = self._renderPass
        swapChainExtent = self._device._swapChainExtent
        graphicsPipeline = self._graphicsPipeline

        allocInfo = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=commandPool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(swapChainFramebuffers),
        )

        commandBuffers = vk.vkAllocateCommandBuffers(device_handle, allocInfo)
        self._commandBuffers = [
            vk.ffi.addressof(commandBuffers, i)[0]
            for i in range(len(swapChainFramebuffers))
        ]

        for i, cmdBuffer in enumerate(self._commandBuffers):
            beginInfo = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
            )

            vk.vkBeginCommandBuffer(cmdBuffer, beginInfo)

            renderPassInfo = vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=renderPass,
                framebuffer=swapChainFramebuffers[i],
                renderArea=[[0, 0], swapChainExtent],
            )

            clearColor = vk.VkClearValue([[0.0, 0.0, 0.0, 1.0]])
            renderPassInfo.clearValueCount = 1
            renderPassInfo.pClearValues = vk.ffi.addressof(clearColor)

            vk.vkCmdBeginRenderPass(cmdBuffer, renderPassInfo, vk.VK_SUBPASS_CONTENTS_INLINE)

            vk.vkCmdBindPipeline(
                cmdBuffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline
            )
            vk.vkCmdDraw(cmdBuffer, 3, 1, 0, 0)

            vk.vkCmdEndRenderPass(cmdBuffer)

            vk.vkEndCommandBuffer(cmdBuffer)

    def _createSemaphores(self):

        device_handle = self._device._handle

        semaphoreInfo = vk.VkSemaphoreCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        )

        self._imageAvailableSemaphore = vk.vkCreateSemaphore(
            device_handle, semaphoreInfo, None
        )
        self._renderFinishedSemaphore = vk.vkCreateSemaphore(
            device_handle, semaphoreInfo, None
        )

    def _drawFrame(self):
        # print("draw", time.time())

        device_handle = self._device._handle
        swapChain = self._device._swapchain
        commandBuffers = self._commandBuffers
        graphicsQueue = self._device._graphicsQueue  # can move this easily
        presentQueue = self._device._presentQueue  # can move this easily
        imageAvailableSemaphore = self._imageAvailableSemaphore
        renderFinishedSemaphore = self._renderFinishedSemaphore

        vkAcquireNextImageKHR = vk.vkGetDeviceProcAddr(device_handle, "vkAcquireNextImageKHR")
        imageIndex = vkAcquireNextImageKHR(
            device_handle,
            swapChain,
            18446744073709551615,  # todo: wat?
            imageAvailableSemaphore,
            vk.VK_NULL_HANDLE,
        )

        submitInfo = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO)

        waitSemaphores = vk.ffi.new("VkSemaphore[]", [imageAvailableSemaphore])
        waitStages = vk.ffi.new(
            "uint32_t[]", [vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
        )
        submitInfo.waitSemaphoreCount = 1
        submitInfo.pWaitSemaphores = waitSemaphores
        submitInfo.pWaitDstStageMask = waitStages

        cmdBuffers = vk.ffi.new("VkCommandBuffer[]", [commandBuffers[imageIndex]])
        submitInfo.commandBufferCount = 1
        submitInfo.pCommandBuffers = cmdBuffers

        signalSemaphores = vk.ffi.new("VkSemaphore[]", [renderFinishedSemaphore])
        submitInfo.signalSemaphoreCount = 1
        submitInfo.pSignalSemaphores = signalSemaphores

        vk.vkQueueSubmit(graphicsQueue, 1, submitInfo, vk.VK_NULL_HANDLE)

        swapChains = [swapChain]
        presentInfo = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=signalSemaphores,
            swapchainCount=len(swapChains),
            pSwapchains=swapChains,
            pImageIndices=[imageIndex],
        )

        vkQueuePresentKHR = vk.vkGetDeviceProcAddr(device_handle, "vkQueuePresentKHR")
        vkQueuePresentKHR(presentQueue, presentInfo)

    def keep_drawing(self):
        # todo: stop when window is closed ...
        async def drawer():
            while True:
                await asyncio.sleep(0.1)
                self._drawFrame()

        asyncio.get_event_loop().create_task(drawer())

##

surface = goovi.Surface()
instance = goovi.Instance(surface=surface)
for d in instance.get_available_devices():
    if d.is_suitable(surface):
        pdevice = d
device = goovi.LogicalDevice(instance, surface, pdevice)

fig = Figure(instance, surface, device)
fig.keep_drawing()
surface.integrate_asyncio()

