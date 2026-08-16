const std = @import("std");
const linux = std.os.linux;
const rl = @import("raylib");

const pixels = @import("pixels.zig");
const protocol = @import("protocol.zig");

pub const ADDRESS = "127.0.0.1";

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    // https://ziglang.org/documentation/0.16.0/std/#std.Io.net.IpAddress.listen
    const addr = try std.Io.net.IpAddress.parseIp4(ADDRESS, protocol.VIDEO_PORT);
    var server = try addr.listen(io, .{
        .reuse_address = true,
        .kernel_backlog = 1, // we only support 1 client for now for simplicity
    });
    defer server.deinit(io);

    const stream = try server.accept(io);
    defer stream.close(io);

    var read_buf: [4096]u8 = undefined;
    var cr = stream.reader(io, &read_buf);
    const r = &cr.interface;

    rl.setTraceLogLevel(.warning);

    rl.initWindow(protocol.FRAME_WIDTH, protocol.FRAME_HEIGHT, "YOLO");
    defer rl.closeWindow();

    rl.setTargetFPS(120);

    // texture_data holds the image data we show on screen.
    const texture_size = protocol.FRAME_WIDTH * protocol.FRAME_HEIGHT * 3;
    const texture_data = try arena.alloc(u8, texture_size);
    @memset(texture_data, 0); // Initialize to black

    const texture_image = rl.Image{
        .data = @ptrCast(texture_data.ptr),
        .width = protocol.FRAME_WIDTH,
        .height = protocol.FRAME_HEIGHT,
        .mipmaps = 1,
        .format = .uncompressed_r8g8b8,
    };
    const texture = try rl.loadTextureFromImage(texture_image);

    var jpeg_buffer = try arena.alloc(u8, protocol.FRAME_BYTES); // NOTE: use worst case buffer for the buffer length.

    while (!rl.windowShouldClose()) {
        var header: protocol.FrameHeader = undefined;
        try r.readSliceAll(std.mem.asBytes(&header));
        try r.readSliceAll(jpeg_buffer[0..header.len]);

        try pixels.jpegToRgb(
            jpeg_buffer[0..header.len],
            texture_data,
            protocol.FRAME_WIDTH,
            protocol.FRAME_HEIGHT,
        );

        rl.updateTexture(texture, @ptrCast(texture_data.ptr));

        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.black);
        rl.drawTexture(texture, 0, 0, rl.Color.white);
    }
}
