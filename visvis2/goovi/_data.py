from ._core import GPUObject


class Sampler(GPUObject):
    """
    A Sampler is not bound to any specific Image. It is rather just a
    set of state parameters, like filtering mode (nearest or linear)
    or addressing mode (repeat, clamp-to-edge, clamp-to-border etc.).
    """


class BufferView(GPUObject):
    """ A BufferView is an object created based on a specific buffer.
    You can pass offset and range during creation to limit the view to
    only a subset of buffer data.
    """


class ImageView(GPUObject):
    """
    An ImageView is a set of parameters referring to a specific image.
    There you can interpret pixels as having some other (compatible)
    format, swizzle any components, and limit the view to a specific
    range of MIP levels or array layers.
    """


class Buffer(GPUObject):
    """
    A Buffer is a container for any binary data that just has its
    length, expressed in bytes.
    """


class Image(GPUObject):
    """
    An Image represents a set of pixels. This is the object known in
    other graphics APIs as a texture. There are many more parameters
    needed to specify creation of an Image. It can be 1D, 2D or 3D,
    have various pixel formats (like R8G8B8A8_UNORM or R32_SFLOAT )
    and can also consist of many discrete images, because it can have
    multiple array layers or MIP levels (or both). Image is a separate
    object type because it doesn’t necessarily consist of just a linear
    set of pixels that can be accessed directly. Images can have a
    different implementation-specific internal format (tiling and
    layout) managed by the graphics driver.
    """


class DeviceMemory(GPUObject):
    """
    Creating a Buffer of certain length or an Image with specific
    dimensions doesn’t automatically allocate memory for it. It is a
    3-step process that must be manually performed by you: Allocate
    DeviceMemory, Create Buffer or Image, Bind them together using
    function vkBindBufferMemory or vkBindImageMemory.

    A DeviceMemory represents a block of memory allocated from a
    specific memory type (as supported by PhysicalDevice) with a
    specific length in bytes. You shouldn’t allocate separate
    DeviceMemory for each Buffer or Image. Instead, you should allocate
    bigger chunks of memory and assign parts of them to your Buffers
    and Images. Allocation is a costly operation and there is a limit
    on maximum number of allocations as well, all of which can be
    queried from your PhysicalDevice.
    """


# The way shaders can access these resources (Buffers, Images and
# Samplers) is through descriptors. Descriptors don’t exist on their
# own, but are always grouped in descriptor sets. But before you create
# a descriptor set, its layout must be specified by creating a
# DescriptorSetLayout, which behaves like a template for a descriptor
# set. Read on at:
# https://gpuopen.com/understanding-vulkan-objects/
