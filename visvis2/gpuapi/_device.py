import vulkan as vk

from ._core import GPUObject, struct_to_dict


deviceExtensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]


class QueueFamilyIndices:
    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0


def findQueueFamilies(instance, surface, device):
    vkGetPhysicalDeviceSurfaceSupportKHR = vk.vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceSurfaceSupportKHR"
    )
    indices = QueueFamilyIndices()
    queueFamilies = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
    for i, queueFamily in enumerate(queueFamilies):
        if (
            queueFamily.queueCount > 0
            and queueFamily.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
        ):
            indices.graphicsFamily = i
        presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface)
        if queueFamily.queueCount > 0 and presentSupport:
            indices.presentFamily = i
        if indices.isComplete():
            break
    return indices


class SwapChainSupportDetails:
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None


def querySwapChainSupport(instance, surface, device):
    details = SwapChainSupportDetails()

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
    )
    details.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface)

    vkGetPhysicalDeviceSurfaceFormatsKHR = vk.vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
    )
    details.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface)

    vkGetPhysicalDeviceSurfacePresentModesKHR = vk.vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
    )
    details.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface)

    return details


# Note that this object does NOT have a handle.
class PhysicalDevice(GPUObject):
    """ Representation of a physical device.
    """

    def __init__(self, instance, ref):
        self._instance = instance
        self._handle = None
        self._ref = ref
        self._props = struct_to_dict(vk.vkGetPhysicalDeviceProperties(ref))
        self._name = self._props["deviceName"]

    def __repr__(self):
        return f"<PhysicalDevice '{self._name}' {self._props['deviceID']}>"

    @property
    def props(self):
        """ The properties of this device.
        """
        return self._props

    @property
    def limits(self):
        """ The limits of this device (shorthand for ``d.props["limits"]``).
        """
        return self._props["limits"]

    def is_suitable(self, surface):
        """ Get whether this device is suitable.
        """
        instance_handle = self._instance._handle
        surface_handle = surface._handle
        device_ref = self._ref

        indices = findQueueFamilies(instance_handle, surface_handle, device_ref)
        extensionsSupported = any(
            extension.extensionName in deviceExtensions
            for extension in vk.vkEnumerateDeviceExtensionProperties(device_ref, None)
        )
        swapChainAdequate = False
        if extensionsSupported:
            swapChainSupport = querySwapChainSupport(
                instance_handle, surface_handle, device_ref
            )
            swapChainAdequate = (not swapChainSupport.formats is None) and (
                not swapChainSupport.presentModes is None
            )
        if indices.isComplete() and extensionsSupported and swapChainAdequate:
            return True


class LogicalDevice(GPUObject):
    def __init__(self):
        pass
