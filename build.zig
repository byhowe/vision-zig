const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_vision_mod = b.createModule(.{
        .root_source_file = b.path("src/vision.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_vision = b.addExecutable(.{
        .name = "vision",
        .root_module = exe_vision_mod,
    });

    const exe_client_mod = b.createModule(.{
        .root_source_file = b.path("src/client.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_client = b.addExecutable(.{
        .name = "client",
        .root_module = exe_client_mod,
    });

    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
        .linkage = .dynamic,
    });

    const raylib = raylib_dep.module("raylib");
    const raylib_artifact = raylib_dep.artifact("raylib");

    const ort = b.addTranslateC(.{
        .root_source_file = b.path("src/ffi/ort.h"),
        .target = target,
        .optimize = optimize,
    });
    ort.linkSystemLibrary("onnxruntime", .{});

    const vl = b.addTranslateC(.{
        .root_source_file = b.path("src/ffi/vl.h"),
        .target = target,
        .optimize = optimize,
    });

    const stb = b.addTranslateC(.{
        .root_source_file = b.path("src/ffi/stb_image.h"),
        .target = target,
        .optimize = optimize,
    });
    const stb_mod = stb.createModule();
    stb_mod.addCSourceFile(.{
        .file = b.path("src/ffi/stb_image.c"),
        .flags = &.{"-O3"},
    });

    exe_vision.root_module.linkLibrary(raylib_artifact);
    exe_vision.root_module.addImport("raylib", raylib);
    exe_vision.root_module.addImport("ort", ort.createModule());
    exe_vision.root_module.addImport("vl", vl.createModule());
    exe_vision.root_module.addImport("stb", stb_mod);

    exe_client.root_module.addImport("raylib", raylib);
    exe_client.root_module.addImport("stb", stb_mod);

    b.installArtifact(exe_vision);
    b.installArtifact(exe_client);
}
