import vulkan as vk

from ._core import GPUObject, struct_to_dict


deviceExtensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]


def chooseSwapSurfaceFormat(availableFormats):
    if (
        len(availableFormats) == 1
        and availableFormats[0].format == vk.VK_FORMAT_UNDEFINED
    ):
        return VkSurfaceFormatKHR(vk.VK_FORMAT_B8G8R8A8_UNORM, 0)

    for availableFormat in availableFormats:
        if (
            availableFormat.format == vk.VK_FORMAT_B8G8R8A8_UNORM
            and availableFormat.colorSpace == 0
        ):
            return availableFormat

    return availableFormats[0]


def chooseSwapPresentMode(availablePresentModes):
    for availablePresentMode in availablePresentModes:
        if availablePresentMode == vk.VK_PRESENT_MODE_MAILBOX_KHR:
            return availablePresentMode

    return vk.VK_PRESENT_MODE_FIFO_KHR


def chooseSwapExtent(capabilities):
    WIDTH, HEIGHT = 640, 480  # todo: get from surface
    width = max(
        capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, WIDTH)
    )
    height = max(
        capabilities.minImageExtent.height,
        min(capabilities.maxImageExtent.height, HEIGHT),
    )
    return vk.VkExtent2D(width, height)



class Device(GPUObject):
    """
    A Device can be thought of as a logical device, or opened device.
    It is the main object that represents an initialized Vulkan device
    that is ready to create all other objects.

    In principal you can create multiple logical devices in the same application.
    """

    def __init__(self, instance, surface, pdevice):
        self._handle = None
        self._swapchain = None
        self._swapChainImageViews = None

        # self._instance = instance
        # self._surface = surface
        self._pdevice = pdevice

        self._handle = self._create_handle(instance, surface, pdevice)
        self._createSwapChain(instance, surface, pdevice)
        self._createSwapChainImageViews()

    def _destroy(self):
        if self._swapChainImageViews:
            for i in self._swapChainImageViews:
                vk.vkDestroyImageView(self._handle, i, None)
        self._swapChainImageViews = None
        if self._swapchain:
            func = vk.vkGetDeviceProcAddr(self._handle, "vkDestroySwapchainKHR")
            if func:
                func(self._handle, self._swapchain, None)
        self._swapchain = None
        if self._handle:
            vk.vkDestroyDevice(self._handle, None)
        self._handle = None

    def _create_handle(self, instance, surface, pdevice):
        indices = pdevice.findQueueFamilies(surface)

        uniqueQueueFamilies = {}.fromkeys(
            (indices.graphicsFamily, indices.presentFamily)
        )

        # Create queue info structs
        queueCreateInfos = []
        for queueFamily in uniqueQueueFamilies:
            queueCreateInfo = vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queueFamily,
                queueCount=1,
                pQueuePriorities=[1.0],
            )
            queueCreateInfos.append(queueCreateInfo)

        # Create divice info struct
        deviceFeatures = vk.VkPhysicalDeviceFeatures()
        createInfo = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            flags=0,
            pQueueCreateInfos=queueCreateInfos,
            queueCreateInfoCount=len(queueCreateInfos),
            pEnabledFeatures=[deviceFeatures],
            enabledExtensionCount=len(deviceExtensions),
            ppEnabledExtensionNames=deviceExtensions,
            enabledLayerCount=len(instance._validation_layers),
            ppEnabledLayerNames=instance._validation_layers,
        )
        device = vk.vkCreateDevice(pdevice._ref, createInfo, None)
        if device is None:
            raise Exception("failed to create logical device!")

        # todo: where to instantiate these?
        # A Queue is an object representing a queue of commands to be
        # executed on the device. All the actual work to be done by the
        # GPU is requested by filling CommandBuffers and submitting them
        # to Queues, using the function vkQueueSubmit . If you have
        # multiple queues like the main graphics queue and a compute
        # queue, you can submit different CommandBuffers to each of them.
        # This way you can enable asynchronous compute, which can lead
        # to a substantial speed up if done right.
        self._graphicsQueue = vk.vkGetDeviceQueue(device, indices.graphicsFamily, 0)
        self._presentQueue = vk.vkGetDeviceQueue(device, indices.presentFamily, 0)
        self._computeQueue = None

        return device

    def _createSwapChain(self, instance, surface, pdevice):
        # todo: Not sure if all this swap chan KHR stuff should be here or in the Surface ...

        surface_handle = surface._handle

        swapChainSupport = pdevice.querySwapChainSupport(surface)

        surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if (
            swapChainSupport.capabilities.maxImageCount > 0
            and imageCount > swapChainSupport.capabilities.maxImageCount
        ):
            imageCount = swapChainSupport.capabilities.maxImageCount

        createInfo = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            flags=0,
            surface=surface_handle,
            minImageCount=imageCount,
            imageFormat=surfaceFormat.format,
            imageColorSpace=surfaceFormat.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        )

        indices = pdevice.findQueueFamilies(surface)
        if indices.graphicsFamily != indices.presentFamily:
            createInfo.imageSharingMode = vk.VK_SHARING_MODE_CONCURRENT
            createInfo.queueFamilyIndexCount = 2
            createInfo.pQueueFamilyIndices = [
                indices.graphicsFamily,
                indices.presentFamily,
            ]
        else:
            createInfo.imageSharingMode = vk.VK_SHARING_MODE_EXCLUSIVE

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform
        createInfo.compositeAlpha = vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        createInfo.presentMode = presentMode
        createInfo.clipped = True

        vkCreateSwapchainKHR = vk.vkGetDeviceProcAddr(
            self._handle, "vkCreateSwapchainKHR"
        )
        self._swapchain = vkCreateSwapchainKHR(self._handle, createInfo, None)
        # The swapchain represents a set of images that can be presented
        # on the Surface, e.g. using double- or triple-buffering. From
        # the swapchain you can query it for the Images it contains.
        # These images already have their backing memory allocated by
        # the system.

        self._swapChainImageFormat = surfaceFormat.format
        self._swapChainExtent = extent

    def _createSwapChainImageViews(self):

        # Create swap-chain images
        vkGetSwapchainImagesKHR = vk.vkGetDeviceProcAddr(
            self._handle, "vkGetSwapchainImagesKHR"
        )
        self._swapChainImages = vkGetSwapchainImagesKHR(self._handle, self._swapchain)

        # Create swap-chain image views
        self._swapChainImageViews = []
        components = vk.VkComponentMapping(
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            vk.VK_COMPONENT_SWIZZLE_IDENTITY,
        )
        subresourceRange = vk.VkImageSubresourceRange(
            vk.VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
        )
        for i, image in enumerate(self._swapChainImages):
            createInfo = vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                flags=0,
                image=image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self._swapChainImageFormat,
                components=components,
                subresourceRange=subresourceRange,
            )
            self._swapChainImageViews.append(
                vk.vkCreateImageView(self._handle, createInfo, None)
            )
