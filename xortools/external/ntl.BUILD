load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "ntl_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

configure_make(
  name = "libntl",
  lib_source = ":ntl_srcs",
  prefix_flag = "",
  args = ["-C", "src/", "-j",
    "GMP_PREFIX=$(pwd)/../../gmp/libgmp/",
    "GMP_INCDIR=$(pwd)/../../gmp/libgmp/include",
    "GMP_LIBDIR=$(pwd)/../../gmp/libgmp/lib",
    "GMP_OPT_INCDIR=-I$(pwd)/../../gmp/libgmp/include",
    "GMP_OPT_LIBDIR=-L$(pwd)/../../gmp/libgmp/lib",
  ],
  out_headers_only = False,
  out_static_libs = ["libntl.a"],
  out_include_dir = "include/",
  out_shared_libs = [],
  out_binaries = [],
  out_lib_dir = "lib/",
  configure_in_place = True,
  visibility = ["//visibility:public"],
  deps = ["@gmp//:libgmp"],
  # do not specify linker tool for it
  # (otherwise, if the libtool from bazel's toolchain is supplied,
  # the build script has problems with passing output file to libtool)
  # see #315
  env = {
      "AR": "",
  },
)

