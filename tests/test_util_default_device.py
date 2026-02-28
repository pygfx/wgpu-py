import sys
import subprocess

import wgpu
from testutils import run_tests
from wgpu.utils.device import DefaultDeviceHelper
import pytest


def test_default_device_is_same():
    helper = DefaultDeviceHelper()

    device1 = helper.get_default_device()
    device2 = helper.get_default_device()
    device3 = helper.get_default_device()

    assert isinstance(device1, wgpu.GPUDevice)
    assert device1 is device2
    assert device1 is device3


def test_default_device_configure_feature_level():
    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device("test", feature_level="core")
    assert helper._adapter_kwargs["feature_level"] == "core"

    helper.preconfigure_default_device("test", feature_level="compatibility")
    assert helper._adapter_kwargs["feature_level"] == "compatibility"

    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", feature_level=42)
    with pytest.raises(ValueError):
        helper.preconfigure_default_device("test", feature_level="shazzbot")

    assert helper._adapter_kwargs["feature_level"] == "compatibility"


def test_default_device_configure_power_preference(caplog):
    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device("test1", power_preference="low-power")
    assert helper._adapter_kwargs["power_preference"] == "low-power"
    assert len(caplog.records) == 0

    helper.preconfigure_default_device("test2", power_preference="high-performance")
    assert helper._adapter_kwargs["power_preference"] == "high-performance"

    assert len(caplog.records) == 1
    assert "by 'test1" in caplog.records[0].message

    helper.preconfigure_default_device("test2", power_preference="high-performance")
    assert helper._adapter_kwargs["power_preference"] == "high-performance"
    assert len(caplog.records) == 1  # no new logs

    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", power_preference=42)
    with pytest.raises(ValueError):
        helper.preconfigure_default_device("test", power_preference="shazzbot")

    assert helper._adapter_kwargs["power_preference"] == "high-performance"


def test_default_device_configure_force_fallback_adapter():
    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device("test", force_fallback_adapter=True)
    assert helper._adapter_kwargs["force_fallback_adapter"] == True  # noqa

    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", force_fallback_adapter="yes")


def test_default_device_configure_canvas():
    # Canvases are ducktyped, so they can be str, which don't match the interface, in which case they are ignored
    canvas1 = object()
    canvas2 = object()

    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device("test", canvas=canvas1)  # ok
    assert helper._adapter_kwargs["canvas"] is canvas1
    helper.preconfigure_default_device("test", canvas=canvas2)  # override + warn
    assert helper._adapter_kwargs["canvas"] is canvas2

    # Get device

    _device = helper.get_default_device()

    # kwargs get cleared, to avoid lingering refs
    assert helper._adapter_kwargs == {}

    with pytest.raises(RuntimeError):
        helper.preconfigure_default_device("test", canvas=canvas1)  # ok


def test_default_device_configure_adapter(caplog):
    adapter1 = wgpu.gpu.request_adapter_sync()
    adapter2 = wgpu.gpu.request_adapter_sync()

    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device("test1", power_preference="low-power")
    assert len(caplog.records) == 0

    helper.preconfigure_default_device("test2", adapter=adapter1)
    assert len(caplog.records) == 1
    msg = caplog.records[-1].message
    assert "setting adapter overrides power_preference" in msg

    helper.preconfigure_default_device("test3", power_preference="low-power")
    assert len(caplog.records) == 2
    msg = caplog.records[-1].message
    assert "setting power_preference is ignored because adapter" in msg

    helper.preconfigure_default_device("test4", adapter=adapter1)
    assert len(caplog.records) == 2

    helper.preconfigure_default_device("test5", adapter=adapter2)
    assert len(caplog.records) == 3
    msg = caplog.records[-1].message
    assert "overrides earlier set " in msg and "test4" in msg

    # Get device

    device = helper.get_default_device()
    assert device.adapter is adapter2


def test_default_device_configure_label():
    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device("test", label="foobar")  # ok
    helper.preconfigure_default_device("test", label="spam")  # override + warn

    assert helper._device_kwargs["label"] == "spam"

    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", label=42)

    # Get device

    device1 = helper.get_default_device()

    assert device1.label == "spam"

    # kwargs get cleared, to avoid lingering refs
    assert helper._device_kwargs == {}

    with pytest.raises(RuntimeError):
        helper.preconfigure_default_device("test", label="too_late")


def test_default_device_configure_required_features(caplog):
    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device(
        "test",
        required_features={
            "timestamp-query",
            "float32-filterable",
        },
    )
    helper.preconfigure_default_device("test", required_features=["shader-f16"])
    helper.preconfigure_default_device("test", required_features={"float32-filterable"})

    # For features, the union operator is applied
    assert helper._device_kwargs["required_features"] == {
        "timestamp-query",
        "float32-filterable",
        "shader-f16",
    }

    # Can also remove features
    helper.preconfigure_default_device(
        "test", required_features={"!float32-filterable"}
    )
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "removes earlier set {'float32-filterable'} from the set" in msg

    assert helper._device_kwargs["required_features"] == {
        "timestamp-query",
        "shader-f16",
    }

    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", required_features=42)
    with pytest.raises(ValueError):
        helper.preconfigure_default_device(
            "test",
            required_features={"foobar"},
        )

    # Get device

    device1 = helper.get_default_device()

    assert device1.features == {"timestamp-query", "shader-f16"}

    with pytest.raises(RuntimeError):
        helper.preconfigure_default_device("test", required_features={"shader-f16"})


def test_default_device_configure_required_limits(caplog):
    helper = DefaultDeviceHelper()

    # Configure

    helper.preconfigure_default_device(
        "test", required_limits={"max-bind-groups": 8, "max-buffer-size": 100}
    )
    helper.preconfigure_default_device(
        "test", required_limits={"max-bindings-per-bind-group": 200}
    )

    helper.preconfigure_default_device(
        "test",
        required_limits={
            "max-buffer-size": 200,
            "max-bindings-per-bind-group": 100,
        },
    )

    # For the limits the min() operator is applied per key
    ref1 = {
        "max-bind-groups": 8,
        "max-buffer-size": 100,
        "max-bindings-per-bind-group": 100,
    }
    ref2 = {
        "max-bind-groups": 8,
        "max-buffer-size": 100,
    }
    assert helper._device_kwargs["required_limits"] == ref1

    # Can also remove a limit
    helper.preconfigure_default_device(
        "test", required_limits={"max-bindings-per-bind-group": None}
    )
    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "removes earlier set {'max-bindings-per-bind-group'} from the dict" in msg
    assert helper._device_kwargs["required_limits"] == ref2

    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", required_limits=42)
    with pytest.raises(TypeError):
        helper.preconfigure_default_device("test", required_limits=set())

    # Get device

    device1 = helper.get_default_device()

    limits_subset = {key: val for key, val in device1.limits.items() if key in ref2}
    assert limits_subset == ref2


def test_default_device_and_exit_should_not_hang():
    # See https://github.com/pygfx/wgpu-py/pull/797

    code = "import wgpu; wgpu.get_default_device(); print('ok')"
    timedout = False
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        timedout = True

    assert not timedout, "timed out!"
    assert result.stdout.strip().endswith("ok"), result.stdout
    assert "traceback" not in result.stderr.lower(), result.stderr


if __name__ == "__main__":
    run_tests(globals())
