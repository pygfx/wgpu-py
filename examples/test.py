import os
import asyncio

import vulkan as vk  # todo: should not need this in examples

from visvis2 import goovi



class Figure:
    """ Wraps stuff ...
    """

    def __init__(self, instance, surface, logicaldevice):
        self._instance = instance
        self._surface = surface
        self._device = logicaldevice



##

surface = goovi.Surface()
instance = goovi.Instance(surface=surface)
surface.integrate_asyncio()
for d in instance.get_available_devices():
    if d.is_suitable(surface):
        pdevice = d
device = goovi.LogicalDevice(instance, surface, pdevice)

fig = Figure(instance, surface, device)
