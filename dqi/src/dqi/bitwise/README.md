# Bitwise C Extensions

This directory demonstrates how to integrate a C extension into the `dqi` package.

## Building

Use `pip install -e .` from the repository root to build the extension in-place so
that running `python dqi/bitwise/main.py` works during development. The extension is
compiled with `-O3` for optimization.

## Adding More Extensions

1. Add your `.c` source file in this directory.
2. Register the extension in the top-level `setup.py` by adding a new `Extension` entry.
3. Re-run `pip install -e .` (or `python setup.py build_ext --inplace`) to rebuild.
