Advanced topics
===============

Immediates
----------

Immediates offer a way to set send a small amount of data to the GPU in the command encoder directly, no need for uniform buffer uploads.
They are restricted to rather small sizes, usually 128 or 265 bytes.

Given an adapter, first determine if it supports immediates::

    >> "immediates" in adapter.features
    True

If immediates are supported, determine the maximum number of bytes that can
be allocated for immediates::

    >> adapter.limits["max-immediate-size"]
    256

You must tell the adapter to create a device that supports immediates,
and you must tell it the number of bytes of immediates that you are using.
Overestimating is okay::

    device = adapter.request_device_sync(
        required_features=["immediates"],
        required_limits={"max-immediate-size": 256},
    )

Creating a immediate data struct in your shader code is similar to the way you would create
a uniform buffer.
The same data can be accessed across all shader stages: vertex, fragment and compute::

    struct Immediates {
        vertex_transform: vec4x4f,
        fragment_color: vec4f,
        pick_position: vec2f,
        frame_counter: u32,
    }
    var<immediate> immediate_data: Immediates;

When creating the pipeline layout for this shader, use ``device.create_pipeline_layout(..., immediate_size)``
to specify the number of bytes of immediate data you are using.

Finally, you set the value of the immediates by using
``encoder.set_immediates(range_offset=0, data=<64 bytes>, data_offset=0, data_size=64)``.
