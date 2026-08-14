const std = @import("std");
const ort = @import("ort");
const vl = @import("vl");
const rl = @import("raylib");

const Video = @import("Video.zig");

const DEVICE = "/dev/video0";
const WIDTH = 640;
const HEIGHT = 480;

const YOLO_WIDTH = 640;
const YOLO_HEIGHT = 640;

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

    // setup memory and buffers

    var memory_info: ?*ort.OrtMemoryInfo = null;
    try checkStatus(api, api.*.CreateCpuMemoryInfo.?(ort.OrtArenaAllocator, ort.OrtMemTypeDefault, &memory_info));
    defer api.*.ReleaseMemoryInfo.?(memory_info);

    // arena allocate buffer for the input tensor. we already know the size from the previous debug prints.
    const input_tensor_len = 1 * 3 * YOLO_HEIGHT * YOLO_WIDTH;
    const input_tensor_data = try arena.alloc(f32, input_tensor_len);
    var input_shape = [_]i64{ 1, 3, YOLO_HEIGHT, YOLO_WIDTH };

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

    var video = try Video.init(DEVICE);
    defer video.deinit();

    try video.setFormat(WIDTH, HEIGHT);
    _ = try video.setFramerate(30); // we don't really care about the actual framerate
    try video.requestBuffers(arena);
    try video.mapBuffers();
    defer video.unmapBuffers() catch {};
    try video.queueBuffers();
    try video.streamon();

    var pfds = [_]std.posix.pollfd{.{
        .fd = video.fd,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};

    // track the actual frame time
    var last_frame_timestmap = std.Io.Clock.awake.now(io).nanoseconds;

    var top: Prediction = .{};

    while (!rl.windowShouldClose()) {
        _ = try std.posix.poll(&pfds, 0);

        if ((pfds[0].revents & std.posix.POLL.IN) != 0) {
            var buf = vl.v4l2_buffer{};
            buf.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = vl.V4L2_MEMORY_MMAP;
            try ioctl(video.fd, vl.VIDIOC_DQBUF, @intFromPtr(&buf));

            const new_frame_timestamp = std.Io.Clock.awake.now(io).nanoseconds;
            const time_elapsed = new_frame_timestamp - last_frame_timestmap;
            last_frame_timestmap = new_frame_timestamp;

            // NOTE: Interesting. the fps is much more erradic when the webcam privacy is on.
            // It fluctuates between 15 fps and 30 fps.
            const fps_estimated = @as(f32, @floatFromInt(std.time.ns_per_s)) / @as(f32, @floatFromInt(time_elapsed));

            try yuyvToRgb(video.buffers[buf.index].ptr, @as([*]u8, texture_data.ptr)[0..texture_size], WIDTH, HEIGHT);
            rl.updateTexture(texture, @ptrCast(texture_data.ptr));

            try ioctl(video.fd, vl.VIDIOC_QBUF, @intFromPtr(&buf));

            preprocessYolo(texture_data[0..texture_size], input_tensor_data, WIDTH, HEIGHT, YOLO_WIDTH, YOLO_HEIGHT);

            // FIXME: the documentation says we need to free these with the allocator. how do we do that?
            var input_name: ?[*]u8 = null;
            try checkStatus(api, api.*.SessionGetInputName.?(session, 0, allocator, &input_name));

            var output_name: ?[*]u8 = null;
            try checkStatus(api, api.*.SessionGetOutputName.?(session, 0, allocator, &output_name));

            // TODO: do we need to create this tensor every time? can we do it once outside the loop and fill the data every frame? experiment...
            var input_tensor: ?*ort.OrtValue = null;
            try checkStatus(api, api.*.CreateTensorWithDataAsOrtValue.?(
                memory_info,
                @ptrCast(input_tensor_data.ptr),
                input_tensor_data.len * @sizeOf(f32),
                @ptrCast(&input_shape[0]),
                input_shape.len,
                ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &input_tensor,
            ));
            defer api.*.ReleaseValue.?(input_tensor);

            var output_tensor: ?*ort.OrtValue = null;
            try checkStatus(api, api.*.Run.?(
                session,
                null, // run_options
                &input_name,
                &input_tensor,
                1,
                &output_name,
                1,
                &output_tensor,
            ));
            defer api.*.ReleaseValue.?(output_tensor);

            var out_ptr: [*c]f32 = null;
            try checkStatus(api, api.*.GetTensorMutableData.?(output_tensor, @ptrCast(&out_ptr)));

            top = getTopPrediction(out_ptr, 8400);

            std.debug.print("fps = {d:.2} | {s}\n", .{ fps_estimated, model_labels[top.class_id] });
        }

        rl.beginDrawing();
        rl.clearBackground(rl.Color.black);

        rl.drawTexture(texture, 0, 0, rl.Color.white);
        rl.drawFPS(10, 10);

        // confidence threshold = 0.35
        if (top.score > 0.35) {
            // convert yolo [cx, cy, w, h] into raylib [x, y, w, h]
            const box_w = top.w;
            const box_h = top.h;
            const box_x = top.cx - (box_w / 2.0);
            const box_y = top.cy - (box_h / 2.0);

            const rect = rl.Rectangle{
                .x = box_x,
                .y = box_y,
                .width = box_w,
                .height = box_h,
            };

            rl.drawRectangleLinesEx(rect, 3.0, rl.Color.lime);

            var label_buffer: [64]u8 = undefined;
            const label_text = std.fmt.bufPrintZ(&label_buffer, "{s}: {d:.2}%", .{ model_labels[top.class_id], top.score * 100.0 }) catch "error";

            // draw background for the text
            const text_size = 20;
            const text_width = rl.measureText(label_text, text_size);
            rl.drawRectangle(@intFromFloat(box_x), @as(i32, @intFromFloat(box_y)) - text_size, text_width + 10, text_size, rl.Color.lime);

            // draw label
            rl.drawText(label_text, @intFromFloat(box_x + 5), @as(i32, @intFromFloat(box_y)) - text_size, text_size, rl.Color.black);
        }

        rl.endDrawing();
    }
}

const Prediction = struct {
    class_id: usize = 0,
    score: f32 = 0.0,

    cx: f32 = 0.0,
    cy: f32 = 0.0,
    w: f32 = 0.0,
    h: f32 = 0.0,
};

fn getTopPrediction(out_ptr: [*c]f32, num_anchors: usize) Prediction {
    var max_score: f32 = 0.0;
    var best_class: usize = 0;
    var best_anchor: usize = 0;

    // NOTE: 0..3 are box coords. 4..83 are classes..

    // find the anchor index with the highest score
    for (4..84) |label_idx| {
        for (0..num_anchors) |anchor_idx| {
            // the layout of the tensor is weird. we have a 80 contigous blocks of size 8400.
            // | 8400 | 8400 | ... 80 times ... | 8400 |
            const flat_idx = (label_idx * num_anchors) + anchor_idx;
            const score = out_ptr[flat_idx];

            if (score > max_score) {
                max_score = score;
                best_class = label_idx - 4;
                best_anchor = anchor_idx;
            }
        }
    }

    // extract the bounding box for the anchor with the highest scored label
    // [cx, cy, w, h]
    const cx_raw = out_ptr[(0 * num_anchors) + best_anchor];
    const cy_raw = out_ptr[(1 * num_anchors) + best_anchor];
    const w_raw = out_ptr[(2 * num_anchors) + best_anchor];
    const h_raw = out_ptr[(3 * num_anchors) + best_anchor];

    return .{
        .class_id = best_class,
        .score = max_score,
        .cx = cx_raw,
        .cy = cy_raw,
        .w = w_raw,
        .h = h_raw,
    };
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
