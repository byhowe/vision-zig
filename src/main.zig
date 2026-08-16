const std = @import("std");
const ort = @import("ort");
const vl = @import("vl");
const rl = @import("raylib");

const pixels = @import("pixels.zig");

const Video = @import("Video.zig");
const Yolo = @import("Yolo.zig");

const DEVICE = "/dev/video0";
const WIDTH = 1280;
const HEIGHT = 720;

const CONFIDENCE_THRESHOLD = 0.35;

const YOLO_W_F32: f32 = @floatFromInt(Yolo.WIDTH);
const YOLO_H_F32: f32 = @floatFromInt(Yolo.HEIGHT);
const HALF_YOLO_W: f32 = YOLO_W_F32 / 2.0;
const HALF_YOLO_H: f32 = YOLO_H_F32 / 2.0;

const W_F32: f32 = @floatFromInt(WIDTH);
const H_F32: f32 = @floatFromInt(HEIGHT);
const HALF_W: f32 = W_F32 / 2.0;
const HALF_H: f32 = H_F32 / 2.0;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    // ONNX

    var model = try Yolo.init(arena);

    // RAYLIB

    // suppress unnecessary information
    rl.setTraceLogLevel(.warning);
    std.debug.print("raylib version = {s}\n", .{rl.RAYLIB_VERSION});

    rl.initWindow(WIDTH, HEIGHT, "YOLO");
    rl.setTargetFPS(120);
    defer rl.closeWindow();

    // texture_data holds the image data we show on screen.
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
    try video.streamon();
    defer video.streamoff() catch {};

    var pfds = [_]std.posix.pollfd{.{
        .fd = video.fd,
        .events = std.posix.POLL.IN,
        .revents = 0,
    }};

    // track the actual frame time
    var last_frame_timestamp = std.Io.Clock.awake.now(io).nanoseconds;
    var real_fps: f32 = 0.0;
    var top: Yolo.Prediction = .{};

    var last_ema_timestamp = last_frame_timestamp;
    var latest_detected: bool = false;
    var latest_target_x: f32 = HALF_W;
    var latest_target_y: f32 = HALF_H;

    var inference_running = false;
    var result_center: [2]f32 = .{ HALF_W, HALF_H };
    var pending_center: [2]f32 = .{ HALF_W, HALF_H };

    // TODO: optimization: instead of filling a cropped version every frame, feed the infer
    // function with the uncropped version and let it fill its internal f32 buffer with the
    // cropped version directly. this is currently just an intermediary buffer.
    const rgb_cropped = try arena.alloc(u8, Yolo.WIDTH * Yolo.HEIGHT * 3);
    @memset(rgb_cropped, 0); // Initialize to black

    var ema = EMA.init(io);
    var center: [2]f32 = .{ HALF_W, HALF_H };

    while (!rl.windowShouldClose()) {
        const current_timestamp = std.Io.Clock.awake.now(io).nanoseconds;
        const dt_ns = current_timestamp - last_ema_timestamp;
        last_ema_timestamp = current_timestamp;
        const dt: f32 = @as(f32, @floatFromInt(dt_ns)) / @as(f32, @floatFromInt(std.time.ns_per_s));

        if (inference_running) {
            if (try model.pollResult()) |pred| {
                top = pred;
                result_center = pending_center;
                inference_running = false;

                latest_detected = top.score > CONFIDENCE_THRESHOLD;
                if (latest_detected) {
                    latest_target_x = result_center[0] - HALF_YOLO_W + top.cx;
                    latest_target_y = result_center[1] - HALF_YOLO_H + top.cy;
                }
            }
        }

        _ = try std.posix.poll(&pfds, 0);

        if ((pfds[0].revents & std.posix.POLL.IN) != 0) {
            const jpeg_buffer = try video.dequeueBuffer();
            try pixels.jpegToRgb(jpeg_buffer, texture_data, WIDTH, HEIGHT);
            try video.queueBuffer(jpeg_buffer); // return to kernel immediately, probably not a huge deal

            // TODO: we may have a setting where we use yuyv or mjpeg. so keep this around for now.
            // try pixels.yuyvToRgb(frame_buffer, @as([*]u8, texture_data.ptr)[0..texture_size], WIDTH, HEIGHT);
            rl.updateTexture(texture, @ptrCast(texture_data.ptr));

            // calculate real fps obtained by the frame arrival times
            const new_frame_timestamp = std.Io.Clock.awake.now(io).nanoseconds;
            const time_elapsed = new_frame_timestamp - last_frame_timestamp;
            last_frame_timestamp = new_frame_timestamp;
            real_fps = @as(f32, @floatFromInt(std.time.ns_per_s)) / @as(f32, @floatFromInt(time_elapsed));
            // NOTE: Interesting. the fps is much more erradic when the webcam privacy is on.
            // It fluctuates between 15 fps and 30 fps.

            if (!inference_running) {
                pending_center = center;

                try pixels.cropRgbFrame(
                    texture_data,
                    WIDTH,
                    HEIGHT,
                    rgb_cropped,
                    Yolo.WIDTH,
                    Yolo.HEIGHT,
                    @intFromFloat(pending_center[0] - HALF_YOLO_W),
                    @intFromFloat(pending_center[1] - HALF_YOLO_H),
                );

                try model.startInfer(rgb_cropped, Yolo.WIDTH, Yolo.HEIGHT);
                inference_running = true;
            }
        }

        ema.update(dt, latest_detected, latest_target_x, latest_target_y);
        center = ema.center();

        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.black);
        rl.drawTexture(texture, 0, 0, rl.Color.white);

        var fps_text_buffer: [64]u8 = undefined;
        const fps_text = std.fmt.bufPrintZ(&fps_text_buffer, "FPS: {d:.2}", .{real_fps}) catch unreachable;
        rl.drawText(fps_text, 10, 10, 20, rl.Color.lime);

        // draw rectangle of where the model is seeing.
        rl.drawRectangleLines(
            @intFromFloat(center[0] - HALF_YOLO_W),
            @intFromFloat(center[1] - HALF_YOLO_H),
            Yolo.WIDTH,
            Yolo.HEIGHT,
            rl.Color.red,
        );

        if (top.score > CONFIDENCE_THRESHOLD) {
            // convert yolo [cx, cy, w, h] into raylib [x, y, w, h]
            const box_w = top.w;
            const box_h = top.h;
            const box_x = top.cx - (box_w / 2.0);
            const box_y = top.cy - (box_h / 2.0);

            const rect = rl.Rectangle{
                .x = box_x + result_center[0] - HALF_YOLO_W,
                .y = box_y + result_center[1] - HALF_YOLO_H,
                .width = box_w,
                .height = box_h,
            };

            rl.drawRectangleLinesEx(rect, 3.0, rl.Color.lime);

            var label_buffer: [64]u8 = undefined;
            const label_text = std.fmt.bufPrintZ(&label_buffer, "{s}: {d:.2}%", .{ Yolo.model_labels[top.class_id], top.score * 100.0 }) catch unreachable;

            // draw background for the text
            const text_size = 20;
            const text_width = rl.measureText(label_text, text_size);

            rl.drawRectangle(
                @intFromFloat(rect.x),
                @as(i32, @intFromFloat(rect.y)) - text_size,
                text_width + 10,
                text_size,
                rl.Color.lime,
            );

            // draw label
            rl.drawText(
                label_text,
                @intFromFloat(rect.x + 5),
                @as(i32, @intFromFloat(rect.y)) - text_size,
                text_size,
                rl.Color.black,
            );
        }
    }
}

const EMA = struct {
    const Self = @This();

    ema_x: f32,
    ema_y: f32,

    search_target_x: f32,
    search_target_y: f32,

    sigma: f32,
    prng: std.Random.DefaultPrng,

    // tuning
    const track_speed: f32 = 4.0;
    const drift_speed: f32 = 2.0;
    const sigma_min: f32 = 1.0;
    const sigma_max: f32 = 400.0;
    const sigma_growth_speed: f32 = 4.0;

    pub fn init(io: std.Io) Self {
        var seed: u64 = undefined;
        io.random(std.mem.asBytes(&seed));

        return .{
            .ema_x = @as(f32, @floatFromInt(WIDTH)) / 2.0,
            .ema_y = @as(f32, @floatFromInt(HEIGHT)) / 2.0,
            .search_target_x = HALF_W,
            .search_target_y = HALF_H,
            .sigma = sigma_max,
            .prng = .init(seed),
        };
    }

    // Returns the mid position of the EMA as f32
    pub fn center(self: *Self) [2]f32 {
        return .{
            std.math.clamp(self.ema_x, HALF_YOLO_W, W_F32 - HALF_YOLO_W),
            std.math.clamp(self.ema_y, HALF_YOLO_H, H_F32 - HALF_YOLO_H),
        };
    }

    pub fn update(self: *Self, dt: f32, detected: bool, target_x: f32, target_y: f32) void {
        if (detected) {
            const alpha = 1.0 - std.math.exp(-dt * track_speed);
            self.ema_x = (alpha * target_x) + ((1.0 - alpha) * self.ema_x);
            self.ema_y = (alpha * target_y) + ((1.0 - alpha) * self.ema_y);

            self.sigma = (alpha * sigma_min) + ((1.0 - alpha) * self.sigma);

            self.search_target_x = self.ema_x;
            self.search_target_y = self.ema_y;
        } else {
            self.sigma = @min(sigma_max, self.sigma * std.math.exp(sigma_growth_speed * dt));

            const dx = self.search_target_x - self.ema_x;
            const dy = self.search_target_y - self.ema_y;
            const dist_sq = (dx * dx) + (dy * dy);

            if (dist_sq < 100.0) {
                var rand = self.prng.random();

                const new_tx = HALF_W + (rand.floatNorm(f32) * self.sigma);
                const new_ty = HALF_H + (rand.floatNorm(f32) * self.sigma);

                self.search_target_x = std.math.clamp(new_tx, HALF_YOLO_W, W_F32 - HALF_YOLO_W);
                self.search_target_y = std.math.clamp(new_ty, HALF_YOLO_H, H_F32 - HALF_YOLO_H);
            }

            const drift_alpha = 1.0 - std.math.exp(-dt * drift_speed);
            self.ema_x = (drift_alpha * self.search_target_x) + ((1.0 - drift_alpha) * self.ema_x);
            self.ema_y = (drift_alpha * self.search_target_y) + ((1.0 - drift_alpha) * self.ema_y);
        }
    }
};
