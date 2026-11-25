"""Test the comment remover in apigen_patcher.py."""

from apigen_patcher import CommentRemover


def dedent(code):
    return code.replace("\n    ", "\n")


def test_comment_remover():
    code = """
    #
    # a comment
    # IDL: some idl spec
    # FIXME: unknown api method
    # FIXME: unknown api property
    # FIXME: unknown api class
    # FIXME: missing check_struct
    # FIXME: new method - only user should remove
    # FIXME: was changed - only user should remove
    """

    p = CommentRemover()
    p.apply(dedent(code))
    code = p.dumps()

    assert code.count("#") == 5

    # These are auto-added by the codegen, so are also auto-removed
    assert "unknown" not in code
    assert "missing check_struct" not in code

    # These are not auto-added, so only the user should remove these
    assert "a comment" in code
    assert "IDL" in code
    assert "new" in code
    assert "was changed" in code


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"Running {func.__name__} ...")
            func()
    print("Done")
