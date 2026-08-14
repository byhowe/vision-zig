const std = @import("std");

pub const ConversionError = error{
    InvalidDimensions,
    BufferTooSmall,
};

// https://gist.github.com/wlhe/fcad2999ceb4a826bd811e9fdb6fe652
pub fn yuyvToRgb(yuyv: []const u8, rgb: []u8, width: usize, height: usize) ConversionError!void {
    const num_pixels = width * height;
    const yuv_size = num_pixels * 2; // YUYV uses 2 bytes per pixel
    const rgb_size = num_pixels * 3; // RGB uses 3 bytes per pixel

    if (num_pixels % 2 != 0) return error.InvalidDimensions;

    if (yuyv.len < yuv_size or rgb.len < rgb_size) return error.BufferTooSmall;

    var i: usize = 0;
    var j: usize = 0;
    while (i < rgb_size and j < yuv_size) : ({
        i += 6;
        j += 4;
    }) {
        const yuv_chunk: *const [4]u8 = yuyv[j..][0..4];
        const rgb_chunk: *[6]u8 = rgb[i..][0..6];

        yuyvToRgbPixel(yuv_chunk, rgb_chunk);
    }
}

fn yuyvToRgbPixel(yuyv: *const [4]u8, rgb: *[6]u8) void {
    const y0: f32 = @floatFromInt(yuyv[0]);
    const u: f32 = @floatFromInt(yuyv[1]);
    const y1: f32 = @floatFromInt(yuyv[2]);
    const y: f32 = @floatFromInt(yuyv[3]);

    const u_adj = u - 128.0;
    const v_adj = y - 128.0;

    const r_uv = 1.4065 * v_adj;
    const g_uv = -0.3455 * u_adj - 0.7169 * v_adj;
    const b_uv = 1.1790 * u_adj;

    // First pixel
    rgb[0] = @intFromFloat(std.math.clamp(y0 + r_uv, 0.0, 255.0));
    rgb[1] = @intFromFloat(std.math.clamp(y0 + g_uv, 0.0, 255.0));
    rgb[2] = @intFromFloat(std.math.clamp(y0 + b_uv, 0.0, 255.0));

    // Second pixel
    rgb[3] = @intFromFloat(std.math.clamp(y1 + r_uv, 0.0, 255.0));
    rgb[4] = @intFromFloat(std.math.clamp(y1 + g_uv, 0.0, 255.0));
    rgb[5] = @intFromFloat(std.math.clamp(y1 + b_uv, 0.0, 255.0));
}
