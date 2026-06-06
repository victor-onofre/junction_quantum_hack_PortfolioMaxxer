from setuptools import setup, Extension

ext_modules = [
    Extension(
        name="dqi.bitwise.c_discrete_math",
        sources=["dqi/bitwise/c_discrete_math.c"],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="dqi.bitwise.gerrymandering.gerrymandering_ext",
        sources=[
            "dqi/bitwise/gerrymandering/gerrymandering_ext.cpp",
            "dqi/bitwise/gerrymandering/bruteforce.cpp",
            "dqi/bitwise/gerrymandering/knapsack.cpp",
            "dqi/bitwise/gerrymandering/probability.cpp",
            "dqi/bitwise/gerrymandering/problem_def.cpp",
        ],
        include_dirs=[
            "dqi/bitwise/gerrymandering",
            "dqi/bitwise/gerrymandering/third_party/argparse-3.2/include",
        ],
        extra_compile_args=["-O3", "-std=c++11"],
    ),
]

setup(ext_modules=ext_modules)
