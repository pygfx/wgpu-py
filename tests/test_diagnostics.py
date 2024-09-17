"""
This tests the diagnostics logic itself. It does not do a tests that *uses* the diagnostics.
"""

import wgpu
from wgpu import _diagnostics
from wgpu._diagnostics import (
    DiagnosticsRoot,
    DiagnosticsBase,
    ObjectTracker,
    dict_to_text,
    int_repr,
)

from testutils import run_tests, can_use_wgpu_lib
from pytest import mark


def dedent(text, n):
    return "\n".join(line[n:] for line in text.split("\n"))


class CustomDiagnosticsRoot(DiagnosticsRoot):
    def __enter__(self):
        _diagnostics.diagnostics = self
        return self

    def __exit__(self, *args):
        _diagnostics.diagnostics = wgpu.diagnostics


class CustomDiagnostics(DiagnosticsBase):
    def __init__(self, name):
        super().__init__(name)
        self.tracker = ObjectTracker()

    def get_dict(self):
        return {k: {"count": v} for k, v in self.tracker.counts.items()}


def test_diagnostics_meta():
    # Test that our custom class does what we expect it to do
    assert isinstance(wgpu.diagnostics, DiagnosticsRoot)
    assert wgpu.diagnostics is _diagnostics.diagnostics

    with CustomDiagnosticsRoot() as custom:
        assert custom is _diagnostics.diagnostics

    assert wgpu.diagnostics is _diagnostics.diagnostics


def test_diagnostics_main():
    with CustomDiagnosticsRoot() as custom:
        d1 = CustomDiagnostics("foo")
        d2 = CustomDiagnostics("bar")

        assert "foo" in repr(custom)
        assert "bar" in repr(custom)
        assert "spam" not in repr(custom)

        assert "foo" in repr(d1)
        assert "bar" in repr(d2)

        # Showing report for one topic

        d1.tracker.increase("FooBar")

        reference1 = """
            ██ foo:

                     count

            FooBar:      1

            ██ bar:

            No data
        """

        assert custom.get_report() == dedent(reference1, 12)

        # Showing report for both topics

        d1.tracker.increase("FooBar")
        d2.tracker.increase("XYZ")

        reference2 = """
            ██ foo:

                     count

            FooBar:      2

            ██ bar:

                  count

            XYZ:      1
        """

        assert custom.get_report() == dedent(reference2, 12)

        # Showing report also for newly added topic

        d3 = CustomDiagnostics("spam")
        assert "spam" in repr(custom)

        d3.tracker.increase("FooBar")
        d3.tracker.increase("FooBar")
        d3.tracker.increase("XYZ")

        reference3 = """
            ██ foo:

                     count

            FooBar:      2

            ██ bar:

                  count

            XYZ:      1

            ██ spam:

                     count

            FooBar:      2
               XYZ:      1
        """

        assert custom.get_report() == dedent(reference3, 12)

        # Can also show one

        reference4 = """
            ██ spam:

                     count

            FooBar:      2
               XYZ:      1
        """

        # Showing report also for newly added backend
        assert d3.get_report() == dedent(reference4, 12)

        # The root dict is a dict that maps topics to the per-topic dicts.
        # So it's a dict of dicts of dicts.
        big_dict = custom.get_dict()
        assert isinstance(big_dict, dict)
        for key, val in big_dict.items():
            assert isinstance(val, dict)
            for k, v in val.items():
                assert isinstance(v, dict)

        # These should not fail
        d3.print_report()
        custom.print_report()


def test_dict_to_text_simple():
    # Note the left justification

    d = {"foo": 123456, "bar": "hi", "spam": 4.12345678}

    reference = """
         foo:  123K
         bar:  hi
        spam:  4.12346
    """
    assert dict_to_text(d) == dedent(reference[1:], 8)


def test_dict_to_text_table():
    # Note the right justification

    d = {
        "foo": {"a": 1, "b": 2, "c": 3.1000000},
        "bar": {"a": 4, "b": 5, "c": 6.123456789123},
    }

    reference = """
              a  b        c

        foo:  1  2      3.1
        bar:  4  5  6.12346
    """
    assert dict_to_text(d) == dedent(reference[1:], 8)

    reference = """
        title   b  a

          foo:  2  1
          bar:  5  4
    """
    assert dict_to_text(d, ["title", "b", "a"]) == dedent(reference[1:], 8)


def test_dict_to_text_justification():
    # Strain the justification

    d = {
        "foobarspameggs": {"aprettylongtitle": 1, "b": "cyan", "c": 3},
        "yo": {"aprettylongtitle": 4, "b": "blueberrycake", "c": 6},
    }

    reference = """
                 title   aprettylongtitle              b  c

        foobarspameggs:                 1           cyan  3
                    yo:                 4  blueberrycake  6
    """

    header = ["title", "aprettylongtitle", "b", "c"]
    assert dict_to_text(d, header) == dedent(reference[1:], 8)


def test_dict_to_text_subdicts():
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
               a  b     c     d    e

         foo:  1  2  opt1:  101  102
                     opt2:  103  104
         bar:  3  4  opt2:  105  106
        spam:  5  6
        eggs:  7  8  opt1:  111  112
                     opt2:  113  114
                     opt3:  115  116
    """

    assert dict_to_text(d) == dedent(reference[1:], 8)


def test_dict_to_text_mix():
    # This covers the option to create sub-rows, covering one case, multiple cases, and zero cases.

    d = {
        "foo": {
            "a": 1,
            "b": 2,
            "c": "simple",
            "z": 42,
        },
        "bar": {"b": 4, "c": {"opt2": {"d": 105, "e": 106}}, "a": 3},
        "spam": {"a": 5, "b": None, "c": {}},
        "eggs": {
            "z": 41,
            "a": 7,
            "c": {
                "opt1": {"d": 111, "e": 112},
                "opt2": {"d": 113, "e": 114},
            },
        },
    }

    reference = """
               a  b   z       c     d    e

         foo:  1  2  42  simple
         bar:  3  4        opt2:  105  106
        spam:  5
        eggs:  7     41    opt1:  111  112
                           opt2:  113  114
    """

    assert dict_to_text(d) == dedent(reference[1:], 8)


def test_object_tracker():
    tracker = ObjectTracker()
    counts = tracker.counts

    tracker.increase("FooBar")
    tracker.increase("FooBar")
    tracker.increase("FooBar")
    tracker.increase("SpamEggs")
    tracker.increase("SpamEggs")
    tracker.increase("SpamEggs")

    assert counts == {"FooBar": 3, "SpamEggs": 3}

    tracker.decrease("FooBar")
    tracker.decrease("FooBar")
    tracker.decrease("FooBar")
    tracker.decrease("SpamEggs")
    tracker.decrease("SpamEggs")

    assert counts == {"FooBar": 0, "SpamEggs": 1}

    tracker.increase("FooBar")
    tracker.increase("SpamEggs")

    assert counts == {"FooBar": 1, "SpamEggs": 2}

    tracker.decrease("FooBar")
    tracker.decrease("SpamEggs")
    tracker.decrease("SpamEggs")

    assert counts == {"FooBar": 0, "SpamEggs": 0}


def test_int_repr():
    assert int_repr(0) == "0"
    assert int_repr(7) == "7"
    assert int_repr(912) == "912"

    assert int_repr(1_000) == "1.00K"
    assert int_repr(1_234) == "1.23K"
    assert int_repr(12_345) == "12.3K"
    assert int_repr(123_456) == "123K"

    assert int_repr(1_000_000) == "1.00M"
    assert int_repr(1_234_000) == "1.23M"
    assert int_repr(12_345_000) == "12.3M"
    assert int_repr(123_456_000) == "123M"

    assert int_repr(1_000_000_000) == "1.00G"
    assert int_repr(1_234_000_000) == "1.23G"
    assert int_repr(12_345_000_000) == "12.3G"
    assert int_repr(123_456_000_000) == "123G"

    assert int_repr(-7) == "-7"
    assert int_repr(-912) == "-912"
    assert int_repr(-1000) == "-1.00K"
    assert int_repr(-12_345) == "-12.3K"
    assert int_repr(-123_456_000) == "-123M"


@mark.skipif(not can_use_wgpu_lib, reason="Needs wgpu lib")
def test_diagnostics_with_backends():
    # Just make sure that it runs without errors

    import wgpu.backends.wgpu_native

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
