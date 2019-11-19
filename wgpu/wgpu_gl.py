import re
import time

from .wgpu import BaseWGPU

import OpenGL.GL as gl
import numpy as np


class Device:
    def __init__(self):
        pass

class Pipeline:
    def __init__(self):
        pass

class Renderpass:
    def __init__(self):
        pass


def _parse_shader_error(error):
    """ Parses a single GLSL error and extracts the linenr and description
    Other GLIR implementations may omit this.
    """
    error = str(error)
    # Nvidia
    # 0(7): error C1008: undefined variable "MV"
    m = re.match(r'(\d+)\((\d+)\)\s*:\s(.*)', error)
    if m:
        return int(m.group(2)), m.group(3)
    # ATI / Intel
    # ERROR: 0:131: '{' : syntax error parse error
    m = re.match(r'ERROR:\s(\d+):(\d+):\s(.*)', error)
    if m:
        return int(m.group(2)), m.group(3)
    # Nouveau
    # 0:28(16): error: syntax error, unexpected ')', expecting '('
    m = re.match(r'(\d+):(\d+)\((\d+)\):\s(.*)', error)
    if m:
        return int(m.group(2)), m.group(4)
    # Other ...
    return None, error


def _get_shader_error(code, errors, indentation=0):
    """Get error and show the faulty line + some context
    Other GLIR implementations may omit this.
    """
    # Init
    results = []
    lines = None
    if code is not None:
        lines = [line.strip() for line in code.split('\n')]

    for error in errors.split('\n'):
        # Strip; skip empy lines
        error = error.strip()
        if not error:
            continue
        # Separate line number from description (if we can)
        linenr, error = _parse_shader_error(error)
        if None in (linenr, lines):
            results.append('%s' % error)
        else:
            results.append('on line %i: %s' % (linenr, error))
            if linenr > 0 and linenr < len(lines):
                results.append('  %s' % lines[linenr - 1])

    # Add indentation and return
    results = [' ' * indentation + r for r in results]
    return '\n'.join(results)


class GlWGPU(BaseWGPU):

    def __init__(self):
        self._device_counter = 0
        self._devices = {}

        self._pipeline_counter = 0
        self._pipelines = {}

        self._renderpass_counter = 0
        self._renderpass = {}


    def request_adapter(self, desc: 'RequestAdapterOptions'):
        """ An adapter is like a device, an implementation of webGPU.
        """
        # desc has power_preference and backends, we ignore both
        return 1

    def adapter_request_device(self, adapter_id: int, desc: 'DeviceDescriptor'):
        """ A device is the logical instantiation of an adapter, through which internal objects are created.
        """
        # desc has extensions and limits, we ignore both
        assert adapter_id == 1
        self._device_counter += 1
        self._devices[self._device_counter] = Device()
        return self._device_counter

    def device_create_shader_module(self, device_id: int, desc: 'ShaderModuleDescriptor'):
        assert device_id > 0
        code = desc["code"]
        # Determine target ...
        if "outColor" in code:
            target = gl.GL_FRAGMENT_SHADER
        else:
            target = gl.GL_VERTEX_SHADER
        # Compile
        handle = gl.glCreateShader(target)
        gl.glShaderSource(handle, code)
        gl.glCompileShader(handle)
        # status = gl.glGetShaderParameter(handle, gl.GL_COMPILE_STATUS)
        errors = gl.glGetShaderInfoLog(handle)
        if errors:#not status:
            errors = errors.decode()
            errormsg = _get_shader_error(code, errors, 4)
            raise RuntimeError("Shader compilation error in %s:\n%s" % (target, errormsg))
        return handle

    def device_create_bind_group_layout(self, device_id: int, desc: 'BindGroupLayoutDescriptor'):
        assert device_id > 0
        bindings_list = desc['bindings']
        assert not bindings_list  # for now
        # todo: uniforms
        return 44

    def device_create_bind_group(self, device_id: int, desc: 'BindGroupDescriptor'):
        assert device_id > 0
        bindings_list = desc['bindings']
        assert not bindings_list  # for now
        # todo: how is this different from layout?
        return 45

        # # Create a buffer
        # buffer_handle = gl.glGenBuffers(1)
        # buffer_target = gl.GL_ARRAY_BUFFER
        # gl.glBindBuffer(buffer_target, buffer_handle)
        # gl.glBufferData(buffer_target, np.array([0, 1, 2, 3], np.int32), gl.GL_DYNAMIC_DRAW)
        # gl.glBindBuffer(buffer_target, buffer_handle)
        # gl.glBindBuffer(buffer_target, 0)

    def device_create_pipeline_layout(self, device_id: int, desc: 'PipelineLayoutDescriptor'):
        assert device_id > 0
        bind_group_layouts = desc["bind_group_layouts"]
        return 50

    def device_create_render_pipeline(self, device_id: int, desc: 'RenderPipelineDescriptor'):
        assert device_id > 0
        layout = desc["layout"]
        assert layout == 50
        # Link the shaders into a program
        vert_shader_handle = desc["vertex_stage"]["module"]
        frag_shader_handle = desc["fragment_stage"]["module"]
        prog_handle = gl.glCreateProgram()
        gl.glAttachShader(prog_handle, vert_shader_handle)
        gl.glAttachShader(prog_handle, frag_shader_handle)
        gl.glLinkProgram(prog_handle)
        # Create pipeline
        self._pipeline_counter += 1
        self._pipelines[self._pipeline_counter] = pipeline = Pipeline()
        pipeline.desc = desc
        pipeline.device_id = device_id
        pipeline.prog_handle = prog_handle

        # All the rest is assumed default for now ...
        # desc = {
        #     "layout": 50,
        #     "vertex_stage": {
        #         "module": 999,
        #         "entry_point": "main"
        #     },
        #     "fragment_stage": {
        #         "module": 999,
        #         "entry_point": "main"
        #     },
        #     "primitive_topology": 3,
        #     "rasterization_state": {
        #         "front_face": 0,
        #         "cull_mode": 0,
        #         "depth_bias": 0,
        #         "depth_bias_slope_scale": 0.0,
        #         "depth_bias_clamp": 0.0
        #     },
        #     "color_states": {
        #         "format": 27,
        #         "alpha_blend": {
        #             "src_factor": 1,
        #             "dst_factor": 0,
        #             "operation": 0
        #         },
        #         "color_blend": {
        #             "src_factor": 1,
        #             "dst_factor": 0,
        #             "operation": 0
        #         },
        #         "write_mask": 15
        #     },
        #     "color_states_length": 1,
        #     "depth_stencil_state": null,
        #     "vertex_input": {
        #         "index_format": 0,
        #         "vertex_buffers": [],
        #         "vertex_buffers_length": 0
        #     },
        #     "sample_count": 1,
        #     "sample_mask": 1,
        #     "alpha_to_coverage_enabled": false
        # }
        return self._pipeline_counter

    def device_create_swap_chain(self, device_id: int, surface_id: int, desc: 'SwapChainDescriptor'):
        assert device_id > 0
        # desc = {'usage': 16, 'format': 27, 'width': 640, 'height': 480, 'present_mode': 1}
        self._devices[device_id].glfw_or_qt_window = surface_id  # for now ...
        self._devices[device_id].width = desc["width"]
        self._devices[device_id].height = desc["height"]
        return device_id  # swap_chain_id == device_id

    def swap_chain_get_next_texture(self, swap_chain_id: int):
        assert swap_chain_id > 0
        # no-op
        return self.create_SwapChainOutput(view_id=53)

    def device_create_command_encoder(self, device_id: int, desc: 'CommandEncoderDescriptor'):
        assert device_id > 0
        return device_id  # encoder_id == device_id

    def command_encoder_begin_render_pass(self, encoder_id: 'Id_CommandBuffer_Dummy', desc: 'RenderPassDescriptor'):
        assert encoder_id > 0
        # desc = {'color_attachments': [{'attachment': 53, 'resolve_target': None, 'load_op': 0, 'store_op': 1, 'clear_color': (1, 1, 0, 1)}], 'color_attachments_length': 1, 'depth_stencil_attachment': None}
        self._renderpass_counter += 1
        self._renderpass[self._renderpass_counter] = renderpass = Renderpass()
        # renderpass.encoder_id
        return self._renderpass_counter

    def render_pass_set_pipeline(self, pass_id: int, pipeline_id: int):
        self._renderpass[pass_id].pipeline_id = pipeline_id

    def render_pass_set_bind_group(self, pass_id: int, index: int, bind_group_id: int, offsets: int, offsets_length: 'uintptr'):
        assert pass_id > 0
        assert bind_group_id == 45

    def render_pass_draw(self, pass_id: int, vertex_count: int, instance_count: int, first_vertex: int, first_instance: int):
        renderpass = self._renderpass[pass_id]
        pipeline = self._pipelines[renderpass.pipeline_id]
        device = self._devices[pipeline.device_id]

        gl.glClearColor(0.7, 0.8, 0.9, 0) # background color
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glViewport(0, 0, device.width, device.height)

        gl.glUseProgram(pipeline.prog_handle)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        # print("drawing at ", time.time())

    def device_get_queue(self, device_id: int):
        assert device_id > 0
        return 56

    def render_pass_end_pass(self, pass_id: int):
        self._renderpass.pop(pass_id, None)

    def command_encoder_finish(self, encoder_id: 'Id_CommandBuffer_Dummy', desc: 'CommandBufferDescriptor'):
        assert encoder_id > 0
        # desc is None?

    def queue_submit(self, queue_id: 'Id_Device_Dummy', command_buffers: int, command_buffers_length: 'uintptr'):
        assert queue_id == 56

    def swap_chain_present(self, swap_chain_id: int):
        device_id = swap_chain_id  # swap_chain_id == device_id
        glfw_or_qt_window = self._devices[device_id].glfw_or_qt_window
        if "qt5" in str(glfw_or_qt_window).lower():
            ctx = glfw_or_qt_window.context()
            surface = ctx.surface()
            glfw_or_qt_window.paintGL()
            ctx.swapBuffers(glfw_or_qt_window)
        elif "qt4" in str(glfw_or_qt_window).lower():
            glfw_or_qt_window.swapBuffers()
        else:
            import glfw
            glfw.swap_buffers(self._devices[device_id].glfw_or_qt_window)
