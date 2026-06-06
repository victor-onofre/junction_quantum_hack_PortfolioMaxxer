package(default_visibility = ["//visibility:public"])

config_setting(
  name = "debug",
  values = {"compilation_mode": "dbg"},
)

COPTS = select({
  "//conditions:default": [
      "-Ofast",
      "-flto",
      "-march=native",
      "-W",
      "-Wall",
  ],
  ":debug": [
    "-O2"
  ],
}) + ["-std=c++20"]

LINKOPTS = [
  "-flto",
  "-lpthread",
]

cc_library(
  name = "libsparse",
  copts = COPTS,
  linkopts = LINKOPTS,
  srcs = [
    "sparse/sparse_matrix.cpp"
  ],
  hdrs = [
    "sparse/sparse_matrix.hpp"
  ],
  includes = ["./"],
  deps = [
    "//src:libutils"
  ]
)

cc_library(
  name = "libutils",
  srcs = [
    "utils.cpp",
  ],
  hdrs = [
    "utils.hpp",
  ],
  includes = ["./"],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name = "libxorsat",
  srcs = [
    "xorsat.cpp",
    "bitmatrix.cpp",
  ],
  hdrs = [
    "xorsat.hpp",
    "bitmatrix.hpp",
  ],
  deps = [
    "//src:libutils",
  ],
  includes = ["./"],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name = "libldpc",
  srcs = [
    "ldpc.cpp",
  ],
  hdrs = [
    "ldpc.hpp",
  ],
  deps = [
    "//src:libutils",
    "//src:libxorsat",
  ],
  includes = ["./"],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "load_decode",
  srcs = ["belief/load_decode_main.cpp"],
  copts = COPTS,
  linkopts = LINKOPTS,
  deps = [
    "@argparse//:argparse",
    "//src:libsparse",
    "//src:libbelief",
    "//src:libutils",
    "//src:libxorsat",
    "//src:libldpc",
  ],
)

cc_binary(
  name = "density",
  srcs = ["belief/density_main.cpp"],
  deps = [
    "@argparse//:argparse",
    "//src:libsparse",
    "//src:libutils",
    ":libbelief"
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name = "libbelief",
  srcs = [
    "belief/density.cpp",
    "belief/decoding.cpp",
  ],
  hdrs = [
    "belief/density.hpp",
    "belief/decoding.hpp",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
  deps = [
    "//src:libsparse",
    "//src:libutils",
    ":libirreg",
    ":libxorsat",
    ":libldpc",
  ],
)

TEST_OPTS = [
        "-fsanitize=address",
        "-fsanitize=undefined",
        "-coverage",
    ]

cc_test(
    name = "belief_test",
    srcs = glob(["belief/*.test.cc"]),
    copts = COPTS + TEST_OPTS,
    linkopts = TEST_OPTS,
    deps = [
        "@gtest",
        "@gtest//:gtest_main",
        ":libbelief",
    ],
)

cc_library(
  name = "libirreg",
  srcs = [
    "irreg/distribution.cpp",
    "irreg/irreg.cpp",
  ],
  hdrs = [
    "irreg/distribution.hpp",
    "irreg/irreg.hpp",
  ],
  includes = [
    "./",
  ],
  deps = [
    "//src:libutils",
    "//src:libsparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "random_irregular",
  copts = COPTS,
  linkopts = LINKOPTS,
  srcs = ["irreg/random_irregular.cpp"],
  deps = [
    "@argparse//:argparse",
    "//src:libsparse",
    "//src:libbelief",
    ":libirreg",
  ],
)

cc_binary(
  name = "random_kXOR",
  srcs = ["irreg/random_kXOR.cpp"],
  deps = [
    "@argparse//:argparse",
    "//src:libsparse",
    "//src:libbelief",
    ":libirreg",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name = "libdesign",
  srcs = ["design/design.cpp"],
  hdrs = ["design/design.hpp"],
  copts = COPTS,
  linkopts = LINKOPTS,
  deps = [
    "//src:libsparse",
    "//src:libutils",
    ":libirreg",
    ":liboptimize",
    ":libbelief",
  ],
)

cc_binary(
  name = "design",
  srcs = ["design/design_main.cpp"] + glob(
    ["deps/boost_optimization_1_85_0/**/*.hpp"]
  ),
  deps = [
    "@argparse//:argparse",
    ":libdesign",
    ":liboptimize"
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "eval",
  srcs = ["design/eval_main.cpp"],
  deps = [
    "@argparse//:argparse",
    ":libdesign",
    ":liboptimize"
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name ="liboptimize",
  srcs = [
    "optimize/anneal.cpp",
    "optimize/problem.cpp",
  ],
  hdrs = [
    "optimize/anneal.hpp",
    "optimize/problem.hpp",
  ],
  deps = [
    ":libsparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "anneal",
  srcs = ["optimize/anneal_main.cpp"],
  deps = [
    "@argparse//:argparse",
    ":libsparse",
    ":liboptimize",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "chunk_anneal",
  srcs = ["optimize/chunk_anneal_main.cpp"],
  deps = [
    "@argparse//:argparse",
    ":libsparse",
    ":liboptimize",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "irreg_anneal",
  srcs = ["optimize/irreg_anneal_main.cpp"],
  deps = [
    "@argparse//:argparse",
    ":libsparse",
    ":liboptimize",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "strip_high_degree_clauses",
  srcs = ["optimize/strip_high_degree_clauses_main.cpp"],
  deps = [
    "@argparse//:argparse",
    ":libsparse",
    ":liboptimize",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name = "libgmpwrap",
  srcs = [
    "lattice/gmpwrap.hpp",
    "lattice/gmpwrap.cpp",
  ],
  deps = [
    "@gmp//:libgmp",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "llldemo",
  srcs = [
    "lattice/llldemo.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    "@ntl//:libntl",
    "@gmp//:libgmp",
    ":libsparse",
    ":libgmpwrap",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "lll",
  srcs = [
    "lattice/lll.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    "@ntl//:libntl",
    "@gmp//:libgmp",
    ":libsparse",
    ":libgmpwrap"
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "lll_perp",
  srcs = [
    "lattice/lll_perp.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    "@ntl//:libntl",
    "@gmp//:libgmp",
    ":libsparse",
    ":libgmpwrap"
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
  name = "libbleichenbacher_nguyen",
  srcs = [
    "lattice/bleichenbacher_nguyen.cpp",
  ],
  hdrs = [
    "lattice/bleichenbacher_nguyen.hpp",
  ],
  deps = [
    "@ntl//:libntl",
    "@gmp//:libgmp",
    ":libsparse",
    ":libgmpwrap",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "bleichenbacher_nguyen",
  srcs = [
    "lattice/bleichenbacher_nguyen_main.cpp",
  ],
  deps = [
    ":libbleichenbacher_nguyen",
    "@argparse//:argparse",
    ":libsparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "plantrs",
  srcs = [
    "lattice/plantrs.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    ":libbleichenbacher_nguyen",
    ":libsparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "plant_rand",
  srcs = [
    "lattice/plant_rand.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    ":libbleichenbacher_nguyen",
    ":libsparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_binary(
  name = "brute",
  srcs = [
    "brute.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    "//src:libutils",
    "//src:libxorsat",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_test(
    name = "ntl_test",
    srcs = [
      "lattice/ntl.test.cc",
    ],
    copts = COPTS + TEST_OPTS,
    linkopts = TEST_OPTS,
    deps = [
        "@gtest",
        "@gtest//:gtest_main",
        "@ntl//:libntl",
        ":libbleichenbacher_nguyen"
    ],
)

cc_binary(
  name = "random_dense",
  srcs = [
    "dense/random_dense.cpp",
  ],
  deps = [
    "@argparse//:argparse",
    "//src:libsparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_library(
    name = "libmwtf",
    hdrs = ["ldpc/macwilliams_transform.hpp"],
    copts = COPTS + TEST_OPTS,
    linkopts = TEST_OPTS,
    deps = [
        ":libgmpwrap",
    ],
)

cc_binary(
  name = "macwilliams_transform",
  srcs = ["ldpc/macwilliams_transform_main.cpp"],
  deps = [
    ":libmwtf",
    "@argparse//:argparse",
  ],
  copts = COPTS,
  linkopts = LINKOPTS,
)

cc_test(
    name = "libmwtf_test",
    srcs = ["ldpc/macwilliams_transform.test.cc"],
    copts = COPTS + TEST_OPTS,
    linkopts = TEST_OPTS,
    deps = [
        "@gtest",
        "@gtest//:gtest_main",
        ":libmwtf",
    ],
)
