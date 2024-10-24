import gc

import wgpu.gui
from testutils import run_tests, is_pypy


def test_weakbind():
    weakbind = wgpu.gui._gui_utils.weakbind

    xx = []

    class Foo:
        def bar(self):
            xx.append(1)

    f1 = Foo()
    f2 = Foo()

    b1 = f1.bar
    b2 = weakbind(f2.bar)

    assert len(xx) == 0
    b1()
    assert len(xx) == 1
    b2()
    assert len(xx) == 2

    del f1
    del f2

    if is_pypy:
        gc.collect()

    assert len(xx) == 2
    b1()
    assert len(xx) == 3  # f1 still exists
    b2()
    assert len(xx) == 3  # f2 is gone!


if __name__ == "__main__":
    run_tests(globals())
