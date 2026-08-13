const std = @import("std");
const onnx = @import("onnx");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    _ = io;
    _ = arena;

    std.debug.print("Hello, World!\n", .{});
}
