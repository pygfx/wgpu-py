import os
import sys
import logging

from .._coreutils import get_resource_filename, logger_set_level_callbacks

from cffi import FFI, __version_info__ as cffi_version_info


logger = logging.getLogger("wgpu")  # noqa


if cffi_version_info < (1, 10):  # no-cover
    raise ImportError(f"{__name__} needs cffi 1.10 or later.")


def get_wgpu_header():
    """Read header file and strip some stuff that cffi would stumble on."""
    return _get_wgpu_header(
        get_resource_filename("webgpu.h"),
        get_resource_filename("wgpu.h"),
    )


def _get_wgpu_header(*filenames):
    """Func written so we can use this in both rs_ffi.py and codegen/hparser.py"""
    # Read files
    lines1 = []
    for filename in filenames:
        with open(filename) as f:
            lines1.extend(f.readlines())
    # Deal with pre-processor commands, because cffi cannot handle them.
    # Just removing them, plus a few extra lines, seems to do the trick.
    lines2 = []
    for line in lines1:
        if line.startswith("#"):
            continue
        elif 'extern "C"' in line:
            continue
        line = line.replace("WGPU_EXPORT ", "")
        lines2.append(line)
    return "".join(lines2)


def get_wgpu_lib_path():
    """Get the path to the wgpu library, taking into account the
    WGPU_LIB_PATH environment variable.
    """

    # If path is given, use that or fail trying
    override_path = os.getenv("WGPU_LIB_PATH", "").strip()
    if override_path:
        return override_path

    # Load the debug binary if requested
    debug_mode = os.getenv("WGPU_DEBUG", "").strip() == "1"
    build = "debug" if debug_mode else "release"

    # Get lib filename for supported platforms
    if sys.platform.startswith("win"):  # no-cover
        lib_filename = f"wgpu_native-{build}.dll"
    elif sys.platform.startswith("darwin"):  # no-cover
        lib_filename = f"libwgpu_native-{build}.dylib"
    elif sys.platform.startswith("linux"):  # no-cover
        lib_filename = f"libwgpu_native-{build}.so"
    else:  # no-cover
        raise RuntimeError(
            f"No WGPU library shipped for platform {sys.platform}. Set WGPU_LIB_PATH instead."
        )

    # Note that this can be a false positive, e.g. ARM linux.
    embedded_path = get_resource_filename(lib_filename)
    if not os.path.isfile(embedded_path):  # no-cover
        raise RuntimeError(
            f"Could not find WGPU library in {embedded_path}. {_maybe_get_hint_on_download_script()}"
        )
    else:
        return embedded_path


def _maybe_get_hint_on_download_script():
    root_dir = os.path.join(get_resource_filename(""), "..", "..")
    filename = os.path.abspath(os.path.join(root_dir, "download-wgpu-native.py"))
    uses_repo = os.path.isfile(filename)

    uses_custom_lib = os.getenv("WGPU_LIB_PATH", "").strip()

    if uses_repo and not uses_custom_lib:
        return "You may need to run download-wgpu-native.py (in the root of the repo)."
    return ""


# Configure cffi and load the dynamic library
# NOTE: `import wgpu.backends.rs` is used in pyinstaller tests to verify
# that we can load the DLL after freezing
ffi = FFI()
ffi.cdef(get_wgpu_header())
ffi.set_source("wgpu.h", None)
lib = ffi.dlopen(get_wgpu_lib_path())


def check_expected_version(version_info):
    # Get lib version
    version_int = lib.wgpuGetVersion()
    if version_int < 65536:  # no-cover - old version encoding with 3 ints
        version_info_lib = tuple((version_int >> bits) & 0xFF for bits in (16, 8, 0))
    else:
        version_info_lib = tuple(
            (version_int >> bits) & 0xFF for bits in (24, 16, 8, 0)
        )
    # When the 0.7.0 tag was made, the version was not bumped.
    if version_info_lib == (0, 6, 0, 0):
        version_info_lib = (0, 7, 0)
    # Compare
    if version_info_lib != version_info:  # no-cover
        logger.warning(
            f"Expected wgpu-native version {version_info} but got {version_info_lib}. {_maybe_get_hint_on_download_script()}"
        )


@ffi.callback("void(WGPULogLevel, char *, void *)")
def _logger_callback(level, c_msg, userdata):
    """Called when Rust emits a log message."""
    # Make a copy of the msg. Rust reclaims the memory when this returns
    msg = ffi.string(c_msg).decode(errors="ignore")
    m = {
        lib.WGPULogLevel_Error: logger.error,
        lib.WGPULogLevel_Warn: logger.warning,
        lib.WGPULogLevel_Info: logger.info,
        lib.WGPULogLevel_Debug: logger.debug,
        lib.WGPULogLevel_Trace: logger.debug,
    }
    func = m.get(level, logger.warning)
    func(msg)


def _logger_set_level_callback(level):
    """Called when the log level is set from Python."""
    if level >= 40:
        lib.wgpuSetLogLevel(lib.WGPULogLevel_Error)
    elif level >= 30:
        lib.wgpuSetLogLevel(lib.WGPULogLevel_Warn)
    elif level >= 20:
        lib.wgpuSetLogLevel(lib.WGPULogLevel_Info)
    elif level >= 10:
        lib.wgpuSetLogLevel(lib.WGPULogLevel_Debug)
    elif level >= 5:
        lib.wgpuSetLogLevel(lib.WGPULogLevel_Trace)  # extra level
    else:
        lib.wgpuSetLogLevel(lib.WGPULogLevel_Off)


# Connect Rust logging with Python logging (userdata set to null)
lib.wgpuSetLogCallback(_logger_callback, ffi.NULL)
logger_set_level_callbacks.append(_logger_set_level_callback)
_logger_set_level_callback(logger.level)
