const std = @import("std");
const ort = @import("ort");
const vl = @import("vl");
const rl = @import("raylib");

const DEVICE = "/dev/video0";
const WIDTH = 640;
const HEIGHT = 480;

const YOLO_WIDTH = 640;
const YOLO_HEIGHT = 640;

const NUM_BUFFERS = 4;
const TIMEOUT = 2000;

const model_data: []const u8 = @embedFile("yolov8m.onnx");
// in 0: name='images', shape={ 1, 3, 640, 640 }
// out 0: name='output0', shape={ 1, 84, 8400 }

const model_labels = block: {
    @setEvalBranchQuota(100_000);

    const text = @embedFile("labels.txt");

    var it = std.mem.tokenizeAny(u8, text, "\r\n");
    var count = 0;
    while (it.next()) |_| count += 1;

    var arr: [count][]const u8 = undefined;
    it = std.mem.tokenizeAny(u8, text, "\r\n");
    for (&arr) |*item| item.* = it.next().?;

    break :block arr;
};

const Buffer = struct {
    start: []align(std.heap.page_size_min) u8,
    length: usize,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    // ONNX

    const api_base = ort.OrtGetApiBase().?;
    const version = api_base.*.GetVersionString.?();
    std.debug.print("onnx api version = {s}\n", .{version});

    const api = api_base.*.GetApi.?(ort.ORT_API_VERSION);

    var env: ?*ort.OrtEnv = null;
    try checkStatus(api, api.*.CreateEnv.?(ort.ORT_LOGGING_LEVEL_WARNING, "YOLO", &env));
    defer api.*.ReleaseEnv.?(env);

    var session_options: ?*ort.OrtSessionOptions = null;
    try checkStatus(api, api.*.CreateSessionOptions.?(&session_options));
    defer api.*.ReleaseSessionOptions.?(session_options);

    // create session from the model data
    var session: ?*ort.OrtSession = null;
    try checkStatus(api, api.*.CreateSessionFromArray.?(
        env,
        model_data.ptr,
        model_data.len,
        session_options,
        &session,
    ));
    defer api.*.ReleaseSession.?(session);

    var allocator: ?*ort.OrtAllocator = null;
    try checkStatus(api, api.*.GetAllocatorWithDefaultOptions.?(&allocator));




    // RAYLIB

    // suppress unnecessary information
    rl.setTraceLogLevel(.warning);
    std.debug.print("raylib version = {s}\n", .{rl.RAYLIB_VERSION});

    rl.initWindow(WIDTH, HEIGHT, "YOLO");
    rl.setTargetFPS(30);
    defer rl.closeWindow();

    const texture_size = WIDTH * HEIGHT * 3;
    const texture_data = try arena.alloc(u8, texture_size);
    @memset(texture_data, 0); // Initialize to black
    const texture_image = rl.Image{
        .data = @ptrCast(texture_data.ptr),
        .width = WIDTH,
        .height = HEIGHT,
        .mipmaps = 1,
        .format = .uncompressed_r8g8b8,
    };
    const texture = try rl.loadTextureFromImage(texture_image);

    // V4L2

    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        DEVICE,
        .{ .ACCMODE = .RDWR, .CLOEXEC = true },
        0,
    );
    defer std.Io.Threaded.closeFd(fd);

    var fmt = vl.v4l2_format{};
    fmt.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = vl.V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = vl.V4L2_FIELD_NONE;
    try ioctl(fd, vl.VIDIOC_S_FMT, @intFromPtr(&fmt));

    // set fps
    var parm = vl.v4l2_streamparm{};
    parm.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 10;
    try ioctl(fd, vl.VIDIOC_S_PARM, @intFromPtr(&parm));
    // NOTE: driver picks whatever format is available regardless of what we set.
    std.debug.print("driver fps = {d}\n", .{parm.parm.capture.timeperframe.denominator});

    var req = vl.v4l2_requestbuffers{};
    req.count = NUM_BUFFERS;
    req.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = vl.V4L2_MEMORY_MMAP;
    try ioctl(fd, vl.VIDIOC_REQBUFS, @intFromPtr(&req));

    std.debug.print("buffer count = {d}\n", .{req.count});

    var buf: vl.v4l2_buffer = undefined;
    var buffers: [NUM_BUFFERS]Buffer = undefined;

    for (0..NUM_BUFFERS) |i| {
        buf = vl.v4l2_buffer{};
        buf.index = @intCast(i);
        buf.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = vl.V4L2_MEMORY_MMAP;
        try ioctl(fd, vl.VIDIOC_QUERYBUF, @intFromPtr(&buf));

        const ptr = try std.posix.mmap(
            null,
            buf.length,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            fd,
            buf.m.offset,
        );

        buffers[i].start = ptr;
        buffers[i].length = buf.length;

        buf = vl.v4l2_buffer{};
        buf.index = @intCast(i);
        buf.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = vl.V4L2_MEMORY_MMAP;
        try ioctl(fd, vl.VIDIOC_QBUF, @intFromPtr(&buf));
    }

    defer for (0..NUM_BUFFERS) |i| std.posix.munmap(buffers[i].start);

    const ty = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    try ioctl(fd, vl.VIDIOC_STREAMON, @intFromPtr(&ty));

    var pfds = [_]std.posix.pollfd{.{
        .fd = fd,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};

    // track the actual frame time
    var last_frame_timestmap = std.Io.Clock.awake.now(io).nanoseconds;

    while (!rl.windowShouldClose()) {
        _ = try std.posix.poll(&pfds, 0);

        if ((pfds[0].revents & std.posix.POLL.IN) != 0) {
            buf = vl.v4l2_buffer{};
            buf.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = vl.V4L2_MEMORY_MMAP;
            try ioctl(fd, vl.VIDIOC_DQBUF, @intFromPtr(&buf));

            const new_frame_timestamp = std.Io.Clock.awake.now(io).nanoseconds;
            const time_elapsed = new_frame_timestamp - last_frame_timestmap;
            last_frame_timestmap = new_frame_timestamp;

            // NOTE: Interesting. the fps is much more erradic when the webcam privacy is on.
            // It fluctuates between 15 fps and 30 fps.
            const fps_estimated = @as(f32, @floatFromInt(std.time.ns_per_s)) / @as(f32, @floatFromInt(time_elapsed));
            std.debug.print("fps = {d:.2}\n", .{fps_estimated});

            try yuyvToRgb(buffers[buf.index].start, @as([*]u8, texture_data.ptr)[0..texture_size], WIDTH, HEIGHT);
            rl.updateTexture(texture, @ptrCast(texture_data.ptr));

            try ioctl(fd, vl.VIDIOC_QBUF, @intFromPtr(&buf));
        }

        rl.beginDrawing();

        rl.clearBackground(rl.Color.black);
        rl.drawTexture(texture, 0, 0, rl.Color.white);
        rl.drawFPS(10, 10);

        rl.endDrawing();
    }
}

// Convert an RGB frame into CHW f32 normalized array with top-left letterboxing
fn preprocessYolo(rgb: []const u8, tensor: []f32, src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) void {
    // set all pixels to black initially.
    @memset(tensor, 0.0);
    const channel_stride = dst_w * dst_h;

    const max_y = @min(src_h, dst_h);
    const max_x = @min(src_w, dst_w);

    for (0..max_y) |y| {
        for (0..max_x) |x| {
            const src_idx = (y * src_w + x) * 3;
            const dst_idx = (y * dst_w) + x;

            tensor[0 * channel_stride + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 0])) / 255.0;
            tensor[1 * channel_stride + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 1])) / 255.0;
            tensor[2 * channel_stride + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 2])) / 255.0;
        }
    }
}

pub const ConversionError = error{
    InvalidDimensions,
    BufferTooSmall,
};

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

fn ioctl(fd: std.os.linux.fd_t, request: u32, arg: usize) !void {
    var r = std.os.linux.ioctl(fd, request, arg);
    while (std.os.linux.errno(r) == .INTR) r = std.os.linux.ioctl(fd, request, arg);
    const errno = std.os.linux.errno(r);
    if (errno != .SUCCESS) return error.FailedIoctl;
}

fn checkStatus(api: *const ort.OrtApi, status: ?*ort.OrtStatus) !void {
    if (status) |st| {
        const msg = api.*.GetErrorMessage.?(st);
        std.debug.print("ort error: {s}\n", .{msg});
        api.*.ReleaseStatus.?(st);
        return error.OrtError;
    }
}
