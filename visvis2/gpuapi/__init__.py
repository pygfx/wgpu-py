"""
gpuapi

A Pythonic, object oriented GPU api, similar to vispy.gloo.
This code uses the Vulkan API but exposes an API free of Vulkan specifics,
so that it can hopefully eventually hook into e.g. WebGPU.
"""

from ._instance import Instance
from ._surface import Surface
from ._device import PhysicalDevice
