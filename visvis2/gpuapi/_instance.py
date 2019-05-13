import vulkan as vk
from __main__ import __name__ as default_app_name

from ._core import GPUObject, struct_to_dict
from ._device import PhysicalDevice

version_info = (0, 0, 1)  # of visvis2


class Instance(GPUObject):
    def __init__(self, *, surface=None, extensions=None, validation_layers="default"):

        # Check validation layers. The default is to use the first available.
        # A good default is VK_LAYER_LUNARG_standard_validation, but it is not
        # always available. E.g. my HP laptop has VK_LAYER_VALVE_steam_overlay
        # instead.
        available_validation_layers = self.get_available_validation_layers()
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

        # Make sure that extensions are str
        extensions = [str(x) for x in (extensions or [])]

        # If we are given a surface, add its extensions.
        # NOTE: don't store the surface object to avoid circular refs!
        if surface:
            extensions += surface.get_required_extensions()

        # Add extension that we need if we use validation layers
        if validation_layers:
            extensions.append(vk.VK_EXT_DEBUG_REPORT_EXTENSION_NAME)

        # Create handle
        self._handle = self._create_handle(extensions, validation_layers)
        if surface:
            surface._create_surface(self)

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

    def get_available_validation_layers(self):
        """ Get a list of strings representing the available validation layers
        for the current Vulkan API.
        """
        # Aparently the VK_LAYER_LUNARG_standard_validation is a good default
        # https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
        layer_refs = vk.vkEnumerateInstanceLayerProperties()
        return [layerProperties.layerName for layerProperties in layer_refs]

    def get_available_extensions(self):
        """ Get a list of the available Vulkan extensions for this system.
        """
        extension_refs = vk.vkEnumerateInstanceExtensionProperties(None)
        return [x.extensionName for x in extension_refs]

    def get_available_devices(self):
        """ Get a list of the available devices present on this system.
        Each device is represented as a dict.
        """
        device_refs = vk.vkEnumeratePhysicalDevices(self._handle)
        return [PhysicalDevice(self, d) for d in device_refs]
