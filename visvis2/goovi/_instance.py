import vulkan as vk
from __main__ import __name__ as default_app_name

from ._core import GPUObject, struct_to_dict

version_info = (0, 0, 1)  # of visvis2


# Extensions that will always be loaded
_standard_extensions = []


def register_standard_extension(ext_name):
    """ Intended for surface backends to specify the extensions that they need.
    """
    _standard_extensions.append(ext_name)


def get_available_validation_layers():
    """ Get a list of strings representing the available validation layers
    for the current Vulkan API.
    """
    # Aparently the VK_LAYER_LUNARG_standard_validation is a good default
    # https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
    layer_refs = vk.vkEnumerateInstanceLayerProperties()
    return [layerProperties.layerName for layerProperties in layer_refs]


def get_available_extensions():
    """ Get a list of the available Vulkan extensions for this system.
    """
    extension_refs = vk.vkEnumerateInstanceExtensionProperties(None)
    return [x.extensionName for x in extension_refs]


class Instance(GPUObject):
    """
    An Instance represents the connection from your application to
    the Vulkan runtime and therefore only should exist once in your
    application. It also stores all application specific state required
    to use Vulkan.

    It is used for general configuration, such as listing/specifying
    validation layers, extensions, and available (physical) devices.
    """

    def __init__(self, extensions=None, validation_layers="default"):

        # Check validation layers. The default is to use the first available.
        # A good default is VK_LAYER_LUNARG_standard_validation, but it is not
        # always available. E.g. my HP laptop has VK_LAYER_VALVE_steam_overlay
        # instead.
        available_validation_layers = get_available_validation_layers()
        if validation_layers is None:
            validation_layers = []
        elif validation_layers == "default":
            if available_validation_layers:
                validation_layers = [available_validation_layers[0]]
            else:
                validation_layers = []
        elif validation_layers == "all":
            validation_layers = list(available_validation_layers)
        else:
            for layer in validation_layers:
                if layer not in available_validation_layers:
                    raise ValueError(f"validation layer {layer} not available!")
        self._validation_layers = validation_layers

        # Collect extensions
        all_extensions = []
        if validation_layers:
            all_extensions.append(vk.VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        all_extensions.extend(_standard_extensions)
        all_extensions.extend([str(x) for x in (extensions or [])])

        # Create handle
        self._handle = self._create_handle(all_extensions, validation_layers)

        # Install debug callback for validation output
        if validation_layers:
            self._create_debug_callback()

    def _destroy(self):
        if self._dbcallback and self._handle:
            func = vk.vkGetInstanceProcAddr(
                self._handle, "vkDestroyDebugReportCallbackEXT"
            )
            if func:
                func(self._handle, self._dbcallback, None)

        if self._handle:
            vk.vkDestroyInstance(self._handle, None)

        self._dbcallback = None
        self._handle = None

    def _create_handle(self, extensions, validation_layers):

        # Create an application info struct
        appInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=default_app_name,
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="visvis2",
            engineVersion=vk.VK_MAKE_VERSION(*version_info),  # visvis 2 version
            apiVersion=vk.VK_MAKE_VERSION(1, 0, 3),  # Vulkan API version
        )

        # Create the instance info struct
        createInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=appInfo,
            enabledExtensionCount=len(extensions),
            ppEnabledExtensionNames=extensions,
            enabledLayerCount=len(validation_layers),
            ppEnabledLayerNames=validation_layers,
        )

        # Create the instance!
        return vk.vkCreateInstance(createInfo, None)

    def _create_debug_callback(self):
        def debugCallback(*args):
            print("DEBUG: {} {}".format(args[5], args[6]))
            return 0

        # Create struct
        createInfo = vk.VkDebugReportCallbackCreateInfoEXT(
            sType=vk.VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            flags=vk.VK_DEBUG_REPORT_ERROR_BIT_EXT | vk.VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=debugCallback,
        )

        # Create the callback
        func = vk.vkGetInstanceProcAddr(self._handle, "vkCreateDebugReportCallbackEXT")
        if func:
            self._dbcallback = func(self._handle, createInfo, None)
        else:
            raise Exception("failed to set up debug callback!")

    def get_available_devices(self):
        """ Get a list of the available devices present on this system.
        Each device is represented as a dict.
        """
        device_refs = vk.vkEnumeratePhysicalDevices(self._handle)
        return [PhysicalDevice(self, d) for d in device_refs]


class PhysicalDevice(GPUObject):
    """
    A PhysicalDevice represents a specific Vulkan-compatible device,
    like a graphics card or integrated graphics. This object exposes
    the hardware's vendorID, deviceID, properties, and limits.

    You choose a physical device at the beginning of your program, and use
    it to create a logical device (a Device object), which represents
    the application state for the device.

    """

    # Note that this object does NOT have a handle.

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

    def findQueueFamilies(self, surface):
        instance_handle = self._instance._handle
        pdevice_ref = self._ref
        surface_handle = surface._handle

        vkGetPhysicalDeviceSurfaceSupportKHR = vk.vkGetInstanceProcAddr(
            instance_handle, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )
        indices = QueueFamilyIndices()
        queueFamilies = vk.vkGetPhysicalDeviceQueueFamilyProperties(pdevice_ref)
        for i, queueFamily in enumerate(queueFamilies):
            if (
                queueFamily.queueCount > 0
                and queueFamily.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
            ):
                indices.graphicsFamily = i
            presentSupport = vkGetPhysicalDeviceSurfaceSupportKHR(pdevice_ref, i, surface_handle)
            if queueFamily.queueCount > 0 and presentSupport:
                indices.presentFamily = i
            if indices.isComplete():
                break
        return indices

    def querySwapChainSupport(self, surface):
        instance_handle = self._instance._handle
        pdevice_ref = self._ref
        surface_handle = surface._handle

        details = SwapChainSupportDetails()
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.vkGetInstanceProcAddr(
            instance_handle, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        )
        details.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdevice_ref, surface_handle)

        vkGetPhysicalDeviceSurfaceFormatsKHR = vk.vkGetInstanceProcAddr(
            instance_handle, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        details.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(pdevice_ref, surface_handle)

        vkGetPhysicalDeviceSurfacePresentModesKHR = vk.vkGetInstanceProcAddr(
            instance_handle, "vkGetPhysicalDeviceSurfacePresentModesKHR"
        )
        details.presentModes = vkGetPhysicalDeviceSurfacePresentModesKHR(pdevice_ref, surface_handle)

        return details

    def is_suitable(self, surface):
        """ Get whether this device is suitable for the given surface.
        """
        instance_handle = self._instance._handle
        surface_handle = surface._handle
        device_ref = self._ref

        # todo: get these from the instance?
        deviceExtensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]

        indices = self.findQueueFamilies(surface)
        extensionsSupported = any(
            extension.extensionName in deviceExtensions
            for extension in vk.vkEnumerateDeviceExtensionProperties(device_ref, None)
        )
        swapChainAdequate = False
        if extensionsSupported:
            swapChainSupport = self.querySwapChainSupport(surface)
            swapChainAdequate = (not swapChainSupport.formats is None) and (
                not swapChainSupport.presentModes is None
            )
        if indices.isComplete() and extensionsSupported and swapChainAdequate:
            return True

    # todo: PhysicalDevice can also enumerate Memory Heaps and Memory Types inside them


class SwapChainSupportDetails:
    def __init__(self):
        self.capabilities = None
        self.formats = None
        self.presentModes = None


class QueueFamilyIndices:
    def __init__(self):
        self.graphicsFamily = -1
        self.presentFamily = -1

    def isComplete(self):
        return self.graphicsFamily >= 0 and self.presentFamily >= 0



