const std = @import("std");
const stb = @import("stb");

pub const ConversionError = error{
    InvalidDimensions,
    BufferTooSmall,
};

// https://gist.github.com/wlhe/fcad2999ceb4a826bd811e9fdb6fe652
pub fn yuyvToRgb(yuyv: []const u8, rgb: []u8, width: usize, height: usize) ConversionError!void {
    const num_pixels = width * height;
    const yuv_size = num_pixels * 2;
    const rgb_size = num_pixels * 3;

    if (num_pixels % 2 != 0) return error.InvalidDimensions;
    if (yuyv.len < yuv_size or rgb.len < rgb_size) return error.BufferTooSmall;

    const num_macropixels = num_pixels / 2;

    for (0..num_macropixels) |idx| {
        const yuv_chunk = yuyv[idx * 4 ..][0..4];
        const rgb_chunk = rgb[idx * 6 ..][0..6];

        yuyvToRgbPixel(yuv_chunk, rgb_chunk);
    }
}

inline fn yuyvToRgbPixel(yuyv: *const [4]u8, rgb: *[6]u8) void {
    const y0: i32 = yuyv[0];
    const u: i32 = @as(i32, yuyv[1]) - 128;
    const y1: i32 = yuyv[2];
    const v: i32 = @as(i32, yuyv[3]) - 128;

    // 1.4065 * 256 = 360
    // -0.3455 * 256 = -88
    // -0.7169 * 256 = -184
    // 1.1790 * 256 = 302
    const r_uv = (360 * v) >> 8;
    const g_uv = (-88 * u - 184 * v) >> 8;
    const b_uv = (302 * u) >> 8;

    rgb[0] = fastClamp(y0 + r_uv);
    rgb[1] = fastClamp(y0 + g_uv);
    rgb[2] = fastClamp(y0 + b_uv);

    rgb[3] = fastClamp(y1 + r_uv);
    rgb[4] = fastClamp(y1 + g_uv);
    rgb[5] = fastClamp(y1 + b_uv);
}

inline fn fastClamp(val: i32) u8 {
    return @intCast(@max(0, @min(255, val)));
}

pub const JpegError = error{
    DecodeFailed,
    DimensionMismatch,
    BufferTooSmall,
};

pub fn jpegToRgb(
    jpeg: []const u8,
    rgb: []u8,
    expected_w: usize,
    expected_h: usize,
) JpegError!void {
    var width = 0;
    var height = 0;
    var channels_in_file = 0;
    const desired_channels = 3;

    const decoded_ptr = stb.stbi_load_from_memory(
        jpeg.ptr,
        @intCast(jpeg.len),
        &width,
        &height,
        &channels_in_file,
        desired_channels,
    ) orelse return error.DecodeFailed;
    defer stb.stbi_image_free(decoded_ptr);

    const w: usize = @intCast(width);
    const h: usize = @intCast(height);

    if (w != expected_w or h != expected_h) return error.DimensionMismatch;

    const total_bytes = w * h * desired_channels;
    if (rgb.len < total_bytes) return error.BufferTooSmall;

    @memcpy(rgb[0..total_bytes], decoded_ptr.?[0..total_bytes]);
}
