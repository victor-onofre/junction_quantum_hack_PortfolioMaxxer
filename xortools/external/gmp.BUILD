load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "gmp_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

configure_make(
  name = "libgmp",
  lib_source = ":gmp_srcs",
  configure_options = [],
  args = ["-j"],
  out_headers_only = False,
  out_static_libs = ["libgmp.a"],
  out_include_dir = "include/",
  out_binaries = [],
  out_lib_dir = "lib/",
  # out_include_dir = "include/gmp",
  # shared_libraries = ["libgmp.so"],
  visibility = ["//visibility:public"],
  # do not specify linker tool for it
  # (otherwise, if the libtool from bazel's toolchain is supplied,
  # the build script has problems with passing output file to libtool)
  # see #315
  env = {
    "AR": "",
  },
)
