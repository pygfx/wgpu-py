"""This part is somewhat tricky so we dedicate a test module on it.

The tricky part for implementing the buffer data mapping, as specified by
the WebGPU API, is that its not trivial (at all) to provide an array-like object
for which numpy views can be created, and which we can invalidate in such
a way that the views become un-usable.

We spend the first part of this module demonstrating what techniques
do NOT work. In the second part we demonstrate the working of the used
technique. In the third part we test the buffer mapping API itself.

"""

import ctypes

import numpy as np

from testutils import run_tests
import pytest


ALLOW_SEGFAULTS = False


# %%%%%%%%%%  1. What does NOT work


def test_fail_array_interface_with_ctypes_array():
    # Create an array-like object using __array_interface__,
    # with a ctypes array to hold the data, via a real address pointer.
    # We cannot invalidate the base array-like object.

    class ArrayLike:
        def __init__(self):
            size = 100
            offset = 0
            itemsize = 1
            self._data = (ctypes.c_uint8 * size)()

            readonly = False
            typestr = "<b1"
            ptr = ctypes.addressof(self._data)
            ptr += offset * itemsize

            self.__array_interface__ = dict(
                version=3,
                shape=(size,),
                typestr=typestr,
                descr=[("", typestr)],
                data=(ptr, readonly),
                strides=None,
            )

        def __len__(self):
            shape = self.__array_interface__["shape"]
            return shape[0] if shape else 0

    a = ArrayLike()

    # Create numpy view
    b = np.asarray(a)
    assert b.base is a

    b.dtype = np.int16
    b.shape = 25, 2

    # Set some data
    b[:, 1] = 1

    # Yes, this changes the original
    assert list(a._data)[:8] == [0, 0, 1, 0, 0, 0, 1, 0]

    # Try to invalidate the array
    a.__array_interface__["shape"] = ()
    a.__array_interface__["data"] = None

    # Can still set the data
    b[:, 1] = 2

    # And it has changed the data. If the data would have been really
    # released (by the process) this'd likely caused a segfault.
    assert list(a._data)[:8] == [0, 0, 2, 0, 0, 0, 2, 0]


def test_fail_but_arrays_can_be_resized_so_how_does_that_work():
    # Arrays can be resized, so how does that work with views?
    # Arrays cannot be resized when they have views on them.

    # Create an array
    a = np.array([1, 2, 3, 4], np.int32)

    # We can resize it
    a.resize((6,))

    # Create a view on it
    b = a.view()
    assert b.base is a

    # Cannot resize now, because it has a view - it checks back references!
    with pytest.raises(ValueError):
        a.resize((8,))

    # For context, it is possible to override the behavior, but when
    # you now try to use the arrays, it will segfault at some point.
    # a.resize((4, ), refcheck=False)


def test_fail_make_array_readonly():
    # Arrays can be set readonly, how does that affect views on that array?
    # It does not :)

    # Create an array
    a = np.array([1, 2, 3, 4], np.int32)

    # And a view on it
    b = a.view()
    assert b.base is a

    # Can modify the view
    a[0] = 11
    b[1] = 12
    assert list(a[0:2]) == [11, 12]

    # Now set the base array to readonly
    a.flags.writeable = False

    # Indeed cannot set array
    with pytest.raises(ValueError):
        a[0] = 13

    # But can still set the view!
    b[1] = 14
    assert list(a[0:2]) == [11, 14]


def test_fail_memoryview_can_be_released():
    # Memoryviews can be released, how does that affect the views?
    # It does not really affect them.

    # Create a memoryview
    a = memoryview((ctypes.c_uint8 * 100)()).cast("B")

    # Create a view on it, via a cast
    b = a.cast("h")

    # Create another view on it, now a numpy array
    c = np.frombuffer(a, dtype=np.int16)
    # assert c.base is a  -- apparently not, but we check in a moment that its actually a view

    # And again, but via a different API
    d = np.asarray(a)
    d.dtype = np.int16
    # assert d.base is a  -- nope

    # Can change the data, so c and d are indeed views
    a[0] = 1
    b[1] = 2
    c[2] = 3
    d[3] = 4
    assert a.tolist()[:8] == [1, 0, 2, 0, 3, 0, 4, 0]

    # Now release the memory
    a.release()

    # It works on the memoryview itself!
    with pytest.raises(ValueError):
        a[0] = 11

    if ALLOW_SEGFAULTS:
        # But not on the memoryview view
        b[1] = 12

        # And also not on the np array views
        c[2] = 13
        d[3] = 14


def test_cannot_subclass_memoryview():
    # So any solutions in that direction can be discarted.
    with pytest.raises(TypeError):

        class ArrayLike(memoryview):
            pass


def test_fail_memoryview_wrapped_in_array_interface():
    # An array_interface can either point to a memory address or wrap
    # a buffer-like object. I'd swear that I had this setup working in
    # the sense that the views could not write after the memoryview was
    # released, but I cannot reproduce it, so I must have seen something
    # else.

    class ArrayLike:
        def __init__(self, size):
            self._data = memoryview((ctypes.c_uint8 * size)()).cast("B")
            self.__array_interface__ = dict(
                version=3,
                shape=(size,),
                typestr="<b1",
                descr=[("", "<b1")],
                data=self._data,
                strides=None,
            )

        def release(self):
            self.__array_interface__["data"].release()

    a = ArrayLike(100)

    b = np.asarray(a).view(dtype=np.int16)

    # Can change via the view
    b[0] = 7
    assert a._data.tolist()[:2] == [7, 0]

    # Can invalidate
    a.release()

    # But can still write :(
    if ALLOW_SEGFAULTS:
        b[0] = 8


# %%%%%%%%%%  2. This works


# Euhm, I'm starting to feel there is no solution ...


# %%%%%%%%%%  3. Buffer mapping API


if __name__ == "__main__":
    run_tests(globals())
