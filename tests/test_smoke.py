"""Smoke test: the package imports and the toolchain runs."""


def test_src_package_imports() -> None:
    import src

    assert src.__name__ == "src"
