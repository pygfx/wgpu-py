"""
This tests the diagnostics logic itself. It does not do a tests that *uses* the diagnostics.
"""


import wgpu
from wgpu import _diagnostics
from wgpu._diagnostics import (
    Diagnostics,
    BackendDiagnostics,
    ObjectTracker,
    dict_to_text,
)

from testutils import run_tests, can_use_wgpu_lib
from pytest import mark


class CustomBackendDiagnostics(BackendDiagnostics):
    def get_report_dict(self):
        pass


def dedent(text, n):
    return "\n".join(line[n:] for line in text.split("\n"))


class Customdiagnostics(Diagnostics):
    def __enter__(self):
        _diagnostics.diagnostics = self
        return self

    def __exit__(self, *args):
        _diagnostics.diagnostics = wgpu.diagnostics


def test_diagnostics_meta():
    # Test that our custom class does what we expet it to do
    assert isinstance(wgpu.diagnostics, Diagnostics)
    assert wgpu.diagnostics is _diagnostics.diagnostics

    with Customdiagnostics() as custom:
        assert custom is _diagnostics.diagnostics

    assert wgpu.diagnostics is _diagnostics.diagnostics


def test_diagnostics_main():
    class GPUFooBar:
        pass

    with Customdiagnostics() as custom:
        d1 = BackendDiagnostics("foo")
        d2 = BackendDiagnostics("bar")

        # If nothing to report, reports are empty
        assert custom.get_report() == ""

        d1.tracker.increase(GPUFooBar)

        reference1 = """
            Diagnostics for wgpu - foo backend:

                    #py

            FooBar    1
        """

        # Showing report for one backend
        assert custom.get_report() == dedent(reference1, 12)

        d1.tracker.increase(GPUFooBar)
        d2.tracker.increase(GPUFooBar)

        reference2 = """
            Diagnostics for wgpu - foo backend:

                    #py

            FooBar    2

            Diagnostics for wgpu - bar backend:

                    #py

            FooBar    1
        """

        # Showing report for another backend
        assert custom.get_report() == dedent(reference2, 12)

        d3 = BackendDiagnostics("spam")
        d3.tracker.increase(GPUFooBar)
        d3.tracker.increase(GPUFooBar)
        d3.tracker.increase(GPUFooBar)

        reference3 = """
            Diagnostics for wgpu - foo backend:

                    #py

            FooBar    2

            Diagnostics for wgpu - bar backend:

                    #py

            FooBar    1

            Diagnostics for wgpu - spam backend:

                    #py

            FooBar    3
        """

        # Showing report also for newly added backend
        assert custom.get_report() == dedent(reference3, 12)

        # Can also show one

        reference4 = """
            Diagnostics for wgpu - spam backend:

                    #py

            FooBar    3
        """

        # Showing report also for newly added backend
        assert d3.get_report() == dedent(reference4, 12)


def test_dict_to_text_simple():
    d = {
        "foo": {"a": 1, "b": 2, "c": 3.1000000},
        "bar": {"a": 4, "b": 5, "c": 6.123456789123},
    }

    reference = """
        title  a  b        c

          foo  1  2      3.1
          bar  4  5  6.12346
    """
    assert dict_to_text(d, ["title", "a", "b", "c"]) == dedent(reference[1:], 8)

    reference = """
        title  b  a

          foo  2  1
          bar  5  4
    """
    assert dict_to_text(d, ["title", "b", "a"]) == dedent(reference[1:], 8)


def test_dict_to_text_justification():
    d = {
        "foobarspameggs": {"aprettylongtitle": 1, "b": "cyan", "c": 3},
        "yo": {"aprettylongtitle": 4, "b": "blueberrycake", "c": 6},
    }

    reference = """
                 title  aprettylongtitle              b  c

        foobarspameggs                 1           cyan  3
                    yo                 4  blueberrycake  6
    """

    header = ["title", "aprettylongtitle", "b", "c"]
    assert dict_to_text(d, header) == dedent(reference[1:], 8)


def test_dict_to_text_subs():
    # This covers the option to create sub-rows, covering one case, multiple cases, and zero cases.

    d = {
        "foo": {
            "a": 1,
            "b": 2,
            "c": {"opt1": {"d": 101, "e": 102}, "opt2": {"d": 103, "e": 104}},
        },
        "bar": {"a": 3, "b": 4, "c": {"opt2": {"d": 105, "e": 106}}},
        "spam": {"a": 5, "b": 6, "c": {}},
        "eggs": {
            "a": 7,
            "b": 8,
            "c": {
                "opt1": {"d": 111, "e": 112},
                "opt2": {"d": 113, "e": 114},
                "opt3": {"d": 115, "e": 116},
            },
        },
    }

    reference = """
              a  b     c    d    e

         foo  1  2  opt1  101  102
                    opt2  103  104
         bar  3  4  opt2  105  106
        spam  5  6
        eggs  7  8  opt1  111  112
                    opt2  113  114
                    opt3  115  116
    """

    header = ["", "a", "b", "c", "d", "e"]
    assert dict_to_text(d, header) == dedent(reference[1:], 8)


def test_object_tracker():
    class GPUFooBar:
        pass

    class GPUSpamEggs:
        pass

    counts = {}
    tracker = ObjectTracker(counts)

    tracker.increase(GPUFooBar)
    tracker.increase(GPUFooBar)
    tracker.increase(GPUFooBar)
    tracker.increase(GPUSpamEggs)
    tracker.increase(GPUSpamEggs)
    tracker.increase(GPUSpamEggs)

    assert counts == {"FooBar": 3, "SpamEggs": 3}

    tracker.decrease(GPUFooBar)
    tracker.decrease(GPUFooBar)
    tracker.decrease(GPUFooBar)
    tracker.decrease(GPUSpamEggs)
    tracker.decrease(GPUSpamEggs)

    assert counts == {"FooBar": 0, "SpamEggs": 1}

    tracker.increase(GPUFooBar)
    tracker.increase(GPUSpamEggs)

    assert counts == {"FooBar": 1, "SpamEggs": 2}

    tracker.decrease(GPUFooBar)
    tracker.decrease(GPUSpamEggs)
    tracker.decrease(GPUSpamEggs)

    assert counts == {"FooBar": 0, "SpamEggs": 0}


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_diagnostics_with_backends():
    # Just make sure that it runs without errors

    import wgpu.backends.rs

    text = wgpu.diagnostics.get_report()

    assert "Device" in text
    assert "RenderPipeline" in text
    assert "ShaderModule" in text


def test_texture_format_map_is_complete():

    # When texture formats are added, removed, or changed, we must update our
    # map. This test makes sure we don't forget.

    map_keys = set(_diagnostics.texture_format_to_bpp.keys())
    enum_keys = set(wgpu.TextureFormat)

    too_much = map_keys - enum_keys
    missing = enum_keys - map_keys

    assert not too_much
    assert not missing
    assert map_keys == enum_keys  # for good measure


if __name__ == "__main__":
    run_tests(globals())
