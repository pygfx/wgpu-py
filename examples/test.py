import os
import asyncio

import vulkan as vk  # todo: should not need this in examples

from visvis2 import gpuapi


deviceExtensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]


class QueueFamilyIndices:
    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1
    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


def findQueueFamilies(instance, surface, device):
    vkGetPhysicalDeviceSurfaceSupportKHR = vk.vkGetInstanceProcAddr(instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
    indices = QueueFamilyIndices()
    queueFamilies = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
    for i, queueFamily in enumerate(queueFamilies):
        if queueFamily.queueCount > 0 and queueFamily.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
            indices.graphicsFamily = i
        presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface)
        if queueFamily.queueCount > 0 and presentSupport:
            indices.presentFamily = i
        if indices.isComplete():
            break
    return indices


def chooseSwapSurfaceFormat(availableFormats):
    if len(availableFormats) == 1 and availableFormats[0].format == vk.VK_FORMAT_UNDEFINED:
        return VkSurfaceFormatKHR(vk.VK_FORMAT_B8G8R8A8_UNORM, 0)

    for availableFormat in availableFormats:
        if availableFormat.format == vk.VK_FORMAT_B8G8R8A8_UNORM and availableFormat.colorSpace == 0:
            return availableFormat

    return availableFormats[0]


def chooseSwapPresentMode(availablePresentModes):
    for availablePresentMode in availablePresentModes:
        if availablePresentMode == vk.VK_PRESENT_MODE_MAILBOX_KHR:
            return availablePresentMode

    return vk.VK_PRESENT_MODE_FIFO_KHR


def chooseSwapExtent(capabilities):
    WIDTH, HEIGHT = 640, 480  # todo: get from surface
    width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, WIDTH))
    height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, HEIGHT))
    return vk.VkExtent2D(width, height)


class SwapChainSupportDetails:
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None


def querySwapChainSupport(instance, surface, device):
    details = SwapChainSupportDetails()

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.vkGetInstanceProcAddr(instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
    details.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface)

    vkGetPhysicalDeviceSurfaceFormatsKHR = vk.vkGetInstanceProcAddr(instance, 'vkGetPhysicalDeviceSurfaceFormatsKHR')
    details.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface)

    vkGetPhysicalDeviceSurfacePresentModesKHR = vk.vkGetInstanceProcAddr(instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR')
    details.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface)

    return details


class Figure:

    def __init__(self, instance, surface):
        self._actual_instance = instance
        self._instance = instance._handle
        self._surface = surface._handle
        self._init()
        self._initVulkan()

    def __del__(self):
        self._destroy()
        self._init()

    def _init(self):
        self._swapchain = None
        self._device = None
        self._physicalDevice = None  # todo: can I ditch this, its not an "object"
        self._dbcallback = None

    def _destroy(self):
        if self._device:
            vk.vkDestroyDevice(self._device, None)

    def _initVulkan(self):
        self._physicalDevice = self._selectPhysicalDevice()
        self._device = self._createLogicalDevice()
        self._swapchain = self._createSwapChain()
        # self.__createImageViews()
        # self.__createRenderPass()
        # self.__createGraphicsPipeline()
        # self.__createFramebuffers()
        # self.__createCommandPool()
        # self.__createCommandBuffers()
        # self.__createSemaphores()

    def _selectPhysicalDevice(self):
        selected = None
        for device in vk.vkEnumeratePhysicalDevices(self._instance):
            indices = findQueueFamilies(self._instance, self._surface, device)
            extensionsSupported = any(
                extension.extensionName in deviceExtensions
                for extension in vk.vkEnumerateDeviceExtensionProperties(device, None)
            )
            swapChainAdequate = False
            if extensionsSupported:
                swapChainSupport = querySwapChainSupport(self._instance, self._surface, device)
                swapChainAdequate = (not swapChainSupport.formats is None) and (not swapChainSupport.presentModes is None)
            if indices.isComplete() and extensionsSupported and swapChainAdequate:
                selected = device
                break
        if selected is None:
            raise Exception("failed to find a suitable GPU!")
        device_props = vk.vkGetPhysicalDeviceProperties(selected)
        print("selected", device_props.deviceName)
        return selected

    def get_available_devices(self):
        devices = []
        for device in vk.vkEnumeratePhysicalDevices(self._instance):
            device_props = vk.vkGetPhysicalDeviceProperties(device)
            devices.append(struct_to_dict(device_props))
        return devices

    def _createLogicalDevice(self):
        # In principal, we can create multiple logical devices on the same physical device.

        # todo: can/should we cache the queue family info?
        indices = findQueueFamilies(self._instance, self._surface, self._physicalDevice)
        uniqueQueueFamilies = {}.fromkeys((indices.graphicsFamily, indices.presentFamily))

        # Create queue info structs
        queueCreateInfos = []
        for queueFamily in uniqueQueueFamilies:
            queueCreateInfo = vk.VkDeviceQueueCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex=queueFamily,
                queueCount=1,
                pQueuePriorities=[1.0]
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
            enabledLayerCount=len(self._actual_instance._validation_layers),
            ppEnabledLayerNames=self._actual_instance._validation_layers
        )
        device = vk.vkCreateDevice(self._physicalDevice, createInfo, None)
        if device is None:
            raise Exception("failed to create logical device!")

        # todo: where to instantiate these?
        self._graphicsQueue = vk.vkGetDeviceQueue(device, indices.graphicsFamily, 0)
        self._presentQueue = vk.vkGetDeviceQueue(device, indices.presentFamily, 0)

        return device

    def _createSwapChain(self):
        swapChainSupport = querySwapChainSupport(self._instance, self._surface, self._physicalDevice)

        surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats)
        presentMode = chooseSwapPresentMode(swapChainSupport.presentModes)
        extent = chooseSwapExtent(swapChainSupport.capabilities)

        imageCount = swapChainSupport.capabilities.minImageCount + 1
        if swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount:
            imageCount = swapChainSupport.capabilities.maxImageCount

        createInfo = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            flags=0,
            surface=self._surface,
            minImageCount=imageCount,
            imageFormat=surfaceFormat.format,
            imageColorSpace=surfaceFormat.colorSpace,
            imageExtent=extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        )

        indices = findQueueFamilies(self._instance, self._surface, self._physicalDevice)
        if indices.graphicsFamily != indices.presentFamily:
            createInfo.imageSharingMode = vk.VK_SHARING_MODE_CONCURRENT
            createInfo.queueFamilyIndexCount = 2
            createInfo.pQueueFamilyIndices = [indices.graphicsFamily, indices.presentFamily]
        else:
            createInfo.imageSharingMode = vk.VK_SHARING_MODE_EXCLUSIVE

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform
        createInfo.compositeAlpha = vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        createInfo.presentMode = presentMode
        createInfo.clipped = True

        vkCreateSwapchainKHR = vk.vkGetDeviceProcAddr(self._device, 'vkCreateSwapchainKHR')
        self._swapChain = vkCreateSwapchainKHR(self._device, createInfo, None)

        vkGetSwapchainImagesKHR = vk.vkGetDeviceProcAddr(self._device, 'vkGetSwapchainImagesKHR')
        self._swapChainImages = vkGetSwapchainImagesKHR(self._device, self._swapChain)

        self._swapChainImageFormat = surfaceFormat.format
        self._swapChainExtent = extent


##

surface = gpuapi.Surface()
instance = gpuapi.Instance(surface=surface)
surface.integrate_asyncio()

fig = Figure(instance, surface)
