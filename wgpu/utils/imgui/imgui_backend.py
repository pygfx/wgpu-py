from imgui_bundle import imgui
import ctypes
import wgpu
import numpy as np

VERTEX_SHADER_SRC = """
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Uniforms {
    mvp: mat4x4<f32>,
    gamma: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    out.uv = in.uv;
    return out;
}
"""

FRAGMENT_SHADER_SRC = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct Uniforms {
    mvp: mat4x4<f32>,
    gamma: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var s: sampler;
@group(1) @binding(0) var t: texture_2d<f32>;

@fragment
fn main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = in.color * textureSample(t, s, in.uv);
    let corrected_color = pow(color.rgb, vec3<f32>(uniforms.gamma));
    return vec4<f32>(corrected_color, color.a);
}
"""


class ImguiWgpuBackend:
    """Basic integration base class."""

    def __init__(self, device, target_format):
        if not imgui.get_current_context():
            raise RuntimeError(
                "No valid ImGui context. Use imgui.create_context() first and/or "
                "imgui.set_current_context()."
            )

        self._uniform_data = np.zeros(
            (),
            dtype=[
                ("mvp", "float32", (4, 4)),
                ("gamma", "float32"),
                ("__padding", "uint8", (12)),  # padding to 80 bytes
            ],
        )

        self._sampler = None
        self._bind_group = None
        self._texture_bind_group_layout = None

        self._vertex_buffer = None
        self._vertex_buffer_size = 0
        self._index_buffer = None
        self._index_buffer_size = 0

        # Texture management
        self._textures = {}
        self._texture_views = {}

        self.io = imgui.get_io()
        self.io.set_ini_filename("")
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_vtx_offset
        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures

        self._device = device
        self._target_format = target_format

        self._create_device_objects()

        self._pending_cleanup_textures = []

    def register_texture(self, texture_view: wgpu.GPUTextureView) -> imgui.ImTextureRef:
        """Register a custom wgpu.GPUTextureView (which you want to render using Imgui)
        with backend and return a texture reference in Imgui format.
        """
        # get the id so that imgui can display it
        id_texture = ctypes.c_int32(id(texture_view)).value
        # add texture view to the backend so that it can be retrieved for rendering
        self._texture_views[id_texture] = texture_view

        return imgui.ImTextureRef(id_texture)

    def unregister_texture(self, texture_ref: imgui.ImTextureRef):
        """Unregister a previously registered texture reference."""
        self._pending_cleanup_textures.append(texture_ref)

    def _update_texture(self, tex: imgui.ImTextureData):
        if tex.status == imgui.ImTextureStatus.want_create:
            # Create and upload new texture to graphics system
            assert tex.tex_id == 0
            assert tex.format == imgui.ImTextureFormat.rgba32

            # Create texture
            wgpu_tex = self._device.create_texture(
                label="Dear ImGui Texture",
                size=(tex.width, tex.height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )

            wgpu_tex_view = wgpu_tex.create_view()

            # store indentifiers
            # convert to c int32, because imgui requires it
            id_32 = ctypes.c_int32(id(wgpu_tex_view)).value
            tex.set_tex_id(id_32)

            # store backend user data
            # tex.backend_user_data = (wgpu_tex, wgpu_tex_view)
            self._textures[id_32] = wgpu_tex
            self._texture_views[id_32] = wgpu_tex_view

            # Note: Don't set tex.status to ImTextureStatus.ok to let the code fallthrough below.
        if (
            tex.status == imgui.ImTextureStatus.want_create
            or tex.status == imgui.ImTextureStatus.want_updates
        ):
            wgpu_tex = self._textures[tex.tex_id]
            wgpu_tex_view = self._texture_views[tex.tex_id]
            # Update texture data
            assert tex.format == imgui.ImTextureFormat.rgba32

            # Get upload rect
            if tex.status == imgui.ImTextureStatus.want_create:
                upload_x = 0
                upload_y = 0
            else:
                upload_x = tex.update_rect.x
                upload_y = tex.update_rect.y

            upload_w = tex.update_rect.w
            upload_h = tex.update_rect.h

            # Update full texture or selected blocks
            full_data = tex.get_pixels_array()
            offset = (upload_y * tex.width + upload_x) * tex.bytes_per_pixel
            upload_data = full_data[offset:]

            self._device.queue.write_texture(
                {
                    "texture": wgpu_tex,
                    "mip_level": 0,
                    "origin": (upload_x, upload_y, 0),
                },
                upload_data,
                {"offset": 0, "bytes_per_row": tex.width * tex.bytes_per_pixel},
                (upload_w, upload_h, 1),
            )

            # set status to ok
            tex.set_status(imgui.ImTextureStatus.ok)

        if tex.status == imgui.ImTextureStatus.want_destroy and tex.unused_frames > 0:
            wgpu_tex = self._textures[tex.tex_id]
            wgpu_tex_view = self._texture_views[tex.tex_id]

            assert tex.tex_id == ctypes.c_int32(id(wgpu_tex_view)).value

            # Clear identifiers and mark as destroyed
            ImTextureID_Invalid = 0  # noqa
            tex.set_tex_id(ImTextureID_Invalid)
            tex.set_status(imgui.ImTextureStatus.destroyed)

            self._textures.pop(tex.tex_id, None)
            self._texture_views.pop(tex.tex_id, None)

    def _create_device_objects(self):
        vertex_shader_program = self._device.create_shader_module(
            label="triangle_vert", code=VERTEX_SHADER_SRC
        )
        frag_shader_program = self._device.create_shader_module(
            label="triangle_frag", code=FRAGMENT_SHADER_SRC
        )

        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._sampler = self._device.create_sampler(
            label="ImGui sampler",
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.repeat,
            address_mode_v=wgpu.AddressMode.repeat,
            address_mode_w=wgpu.AddressMode.repeat,
        )

        bind_group_layout = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )

        self._texture_bind_group_layout = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
            ]
        )

        bind_groups_layout_entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self._uniform_data.nbytes,
                },
            },
            {
                "binding": 1,
                "resource": self._sampler,
            },
        ]

        vertex_buffer_descriptor = [
            {
                "array_stride": imgui.VERTEX_SIZE,
                "step_mode": wgpu.VertexStepMode.vertex,  # vertex
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 0,
                        "shader_location": 0,
                    },
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 8,
                        "shader_location": 1,
                    },
                    {
                        "format": wgpu.VertexFormat.unorm8x4,
                        "offset": 16,
                        "shader_location": 2,
                    },
                ],
            }
        ]

        self._bind_group = self._device.create_bind_group(
            layout=bind_group_layout,
            entries=bind_groups_layout_entries,
        )

        self._render_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[bind_group_layout, self._texture_bind_group_layout]
            ),
            vertex={
                "module": vertex_shader_program,
                "entry_point": "main",
                "buffers": vertex_buffer_descriptor,
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            fragment={
                "module": frag_shader_program,
                "entry_point": "main",
                "targets": [
                    {
                        "format": self._target_format,
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.src_alpha,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                        },
                    },
                ],
            },
        )

    def _set_render_state(self, draw_data: imgui.ImDrawData):
        # update the uniform buffer (mvp and gamma)

        l = draw_data.display_pos.x  # noqa: E741
        r = draw_data.display_pos.x + draw_data.display_size.x
        t = draw_data.display_pos.y
        b = draw_data.display_pos.y + draw_data.display_size.y

        mvp = np.array(
            [
                [2.0 / (r - l), 0.0, 0.0, 0.0],
                [0.0, 2.0 / (t - b), 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [(r + l) / (l - r), (t + b) / (b - t), 0.5, 1.0],
            ],
            dtype=np.float32,
        )

        self._uniform_data["mvp"] = mvp
        if self._target_format.endswith("srgb"):
            self._uniform_data["gamma"] = 2.2
        else:
            self._uniform_data["gamma"] = 1.0

        self._device.queue.write_buffer(
            self._uniform_buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

    def _update_vertex_buffer(self, draw_data: imgui.ImDrawData):
        # check if we need to recreate the vertex buffer and index buffer
        vtx_count = draw_data.total_vtx_count
        if self._vertex_buffer is None or self._vertex_buffer_size < vtx_count:
            if self._vertex_buffer is not None:
                self._vertex_buffer.destroy()

            self._vertex_buffer_size = (
                vtx_count + 5000
            )  # add some extra space to avoid recreating the buffer too often

            self._vertex_buffer = self._device.create_buffer(
                size=self._vertex_buffer_size * imgui.VERTEX_SIZE,
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )

        idx_count = draw_data.total_idx_count
        if self._index_buffer is None or self._index_buffer_size < idx_count:
            if self._index_buffer is not None:
                self._index_buffer.destroy()

            self._index_buffer_size = (
                idx_count + 10000
            )  # add some extra space to avoid recreating the buffer too often

            self._index_buffer = self._device.create_buffer(
                size=self._index_buffer_size * imgui.INDEX_SIZE,
                usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
            )

        # for commands in draw_data.cmd_lists:
        #     # todo: maybe we can compose the vertex buffer of all the cmd_lists in one buffer and then write it to the gpu once
        #     self._device.queue.write_buffer(
        #         self._vertex_buffer, global_vtx_offset* imgui.VERTEX_SIZE, ctypes.string_at(ctypes.c_void_p(vertex_mem_address), commands.vtx_buffer.size() * imgui.VERTEX_SIZE)
        #     )
        #     index_mem_address = commands.idx_buffer.data_address()
        #     self._device.queue.write_buffer(
        #         self._index_buffer, global_idx_offset* imgui.INDEX_SIZE, ctypes.string_at(ctypes.c_void_p(index_mem_address), commands.idx_buffer.size() * imgui.INDEX_SIZE)
        #     )

        # compose the buffer of all the cmd_lists in one buffer and then write it to the gpu once
        # instead of writing each cmd_list to the gpu in render

        vtx_data = b""
        idx_data = b""

        for commands in draw_data.cmd_lists:
            vtx_data += ctypes.string_at(
                ctypes.c_void_p(commands.vtx_buffer.data_address()),
                commands.vtx_buffer.size() * imgui.VERTEX_SIZE,
            )
            idx_data += ctypes.string_at(
                ctypes.c_void_p(commands.idx_buffer.data_address()),
                commands.idx_buffer.size() * imgui.INDEX_SIZE,
            )

        self._device.queue.write_buffer(
            self._vertex_buffer, 0, vtx_data, 0, len(vtx_data)
        )
        self._device.queue.write_buffer(
            self._index_buffer, 0, idx_data, 0, len(idx_data)
        )

    def render(
        self,
        draw_data: imgui.ImDrawData,
        render_pass: wgpu.GPURenderPassEncoder,
    ):
        """
        Render the imgui draw data with the given render pass.

        Arguments
        ---------
        draw_data : imgui.ImDrawData
            The draw data to render, this is usually obtained by calling ``imgui.get_draw_data()``
        render_pass : wgpu.GPURenderPassEncoder
            The render pass to render the imgui draw data with
        """

        self._clear_pending_textures()

        if draw_data is None:
            return

        display_width, display_height = draw_data.display_size
        fb_width = int(display_width * draw_data.framebuffer_scale.x)
        fb_height = int(display_height * draw_data.framebuffer_scale.y)

        if fb_width <= 0 or fb_height <= 0 or draw_data.cmd_lists_count == 0:
            return

        if draw_data.textures is not None:
            for tex in draw_data.textures:
                if tex.status != imgui.ImTextureStatus.ok:
                    # update tex
                    self._update_texture(tex)

        self._update_vertex_buffer(draw_data)

        # set render state
        self._set_render_state(draw_data)
        render_pass.set_viewport(0, 0, fb_width, fb_height, 0, 1)

        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_vertex_buffer(0, self._vertex_buffer)
        if imgui.INDEX_SIZE == 2:
            index_fmt = wgpu.IndexFormat.uint16
        else:
            index_fmt = wgpu.IndexFormat.uint32
        render_pass.set_index_buffer(self._index_buffer, index_fmt, 0)
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.set_blend_constant((0.0, 0.0, 0.0, 0.0))

        global_vtx_offset = 0
        global_idx_offset = 0

        clip_scale = draw_data.framebuffer_scale
        clip_off = draw_data.display_pos

        for commands in draw_data.cmd_lists:
            # update vertex buffer
            # # todo: maybe we can compose the vertex buffer of all the cmd_lists in one buffer and then write it to the gpu once
            # vertex_mem_address = commands.vtx_buffer.data_address()
            # self._device.queue.write_buffer(
            #     self._vertex_buffer, global_vtx_offset* imgui.VERTEX_SIZE, ctypes.string_at(ctypes.c_void_p(vertex_mem_address), commands.vtx_buffer.size() * imgui.VERTEX_SIZE)
            # )
            # index_mem_address = commands.idx_buffer.data_address()
            # self._device.queue.write_buffer(
            #     self._index_buffer, global_idx_offset* imgui.INDEX_SIZE, ctypes.string_at(ctypes.c_void_p(index_mem_address), commands.idx_buffer.size() * imgui.INDEX_SIZE)
            # )

            for command in commands.cmd_buffer:
                # todo command.user_callback
                tex_id = command.tex_ref.get_tex_id()

                tex_view = self._texture_views[tex_id]

                texture_bind_group = getattr(tex_view, "__imgui_bind_group", None)

                if texture_bind_group is None:
                    texture_bind_group = self._device.create_bind_group(
                        layout=self._texture_bind_group_layout,
                        entries=[
                            {
                                "binding": 0,
                                "resource": tex_view,
                            }
                        ],
                    )
                    # cache the bind group
                    tex_view.__imgui_bind_group = texture_bind_group

                render_pass.set_bind_group(1, texture_bind_group)

                clip_rect = command.clip_rect
                clip_min = [
                    (clip_rect.x - clip_off.x) * clip_scale.x,
                    (clip_rect.y - clip_off.y) * clip_scale.y,
                ]
                clip_max = [
                    (clip_rect.z - clip_off.x) * clip_scale.x,
                    (clip_rect.w - clip_off.y) * clip_scale.y,
                ]
                if clip_min[0] < 0:
                    clip_min[0] = 0
                if clip_min[1] < 0:
                    clip_min[1] = 0
                if clip_max[0] > fb_width:
                    clip_max[0] = fb_width
                if clip_max[1] > fb_height:
                    clip_max[1] = fb_height

                if clip_max[0] - clip_min[0] <= 0 or clip_max[1] - clip_min[1] <= 0:
                    continue

                render_pass.set_scissor_rect(
                    int(clip_min[0]),
                    int(clip_min[1]),
                    int(clip_max[0] - clip_min[0]),
                    int(clip_max[1] - clip_min[1]),
                )

                render_pass.draw_indexed(
                    command.elem_count,
                    1,
                    command.idx_offset + global_idx_offset,
                    command.vtx_offset + global_vtx_offset,
                    0,
                )

            global_vtx_offset += commands.vtx_buffer.size()
            global_idx_offset += commands.idx_buffer.size()

    def _clear_pending_textures(self):
        for tex_ref in self._pending_cleanup_textures:
            tex_id = tex_ref.get_tex_id()
            self._texture_views.pop(tex_id, None)
            self._textures.pop(tex_id, None)
        self._pending_cleanup_textures.clear()
