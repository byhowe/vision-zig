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

    var detector = try Detector.init(arena);
    defer detector.deinit();

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
    var fps = FpsTracker.init(io);

    var last_ema_timestamp = fps.last_timestamp;

    var aim = AimTracker.init(io);
    var crop_center: [2]f32 = .{ HALF_W, HALF_H };

    while (!rl.windowShouldClose()) {
        const now = std.Io.Clock.awake.now(io).nanoseconds;
        const dt_ns = now - last_ema_timestamp;
        last_ema_timestamp = now;
        const dt: f32 = @as(f32, @floatFromInt(dt_ns)) / @as(f32, @floatFromInt(std.time.ns_per_s));

        _ = try detector.collectResult();

        _ = try std.posix.poll(&pfds, 0);

        if ((pfds[0].revents & std.posix.POLL.IN) != 0) {
            const jpeg_buffer = try video.dequeueBuffer();
            try pixels.jpegToRgb(jpeg_buffer, texture_data, WIDTH, HEIGHT);
            try video.queueBuffer(jpeg_buffer); // return to kernel immediately, probably not a huge deal

            // TODO: we may have a setting where we use yuyv or mjpeg. so keep this around for now.
            // try pixels.yuyvToRgb(frame_buffer, @as([*]u8, texture_data.ptr)[0..texture_size], WIDTH, HEIGHT);
            rl.updateTexture(texture, @ptrCast(texture_data.ptr));

            // calculate real fps obtained by the frame arrival times
            fps.tick(io);
            // NOTE: Interesting. the fps is much more erradic when the webcam privacy is on.
            // It fluctuates between 15 fps and 30 fps.

            try detector.submitFrame(texture_data, crop_center);
        }

        aim.update(dt, detector.target_detected, detector.target_center);
        crop_center = aim.center();

        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.black);
        rl.drawTexture(texture, 0, 0, rl.Color.white);

        drawFpsCounter(fps.instant_fps);

        // draw rectangle of where the model is seeing.
        rl.drawRectangleLines(
            @intFromFloat(crop_center[0] - HALF_YOLO_W),
            @intFromFloat(crop_center[1] - HALF_YOLO_H),
            Yolo.WIDTH,
            Yolo.HEIGHT,
            rl.Color.red,
        );

        if (detector.target_detected) {
            drawBoundingBox(detector.last_pred, detector.last_pred_center);
        }
    }
}

const AimTracker = struct {
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

    pub fn update(self: *Self, dt: f32, detected: bool, target: [2]f32) void {
        if (detected) {
            const alpha = 1.0 - std.math.exp(-dt * track_speed);
            self.ema_x = (alpha * target[0]) + ((1.0 - alpha) * self.ema_x);
            self.ema_y = (alpha * target[1]) + ((1.0 - alpha) * self.ema_y);

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

const FpsTracker = struct {
    const Self = @This();

    instant_fps: f32 = 0.0,
    last_timestamp: i96,

    pub fn init(io: std.Io) Self {
        return .{ .last_timestamp = std.Io.Clock.awake.now(io).nanoseconds };
    }

    pub fn tick(self: *Self, io: std.Io) void {
        const new_timestamp = std.Io.Clock.awake.now(io).nanoseconds;
        const dt = @as(f32, @floatFromInt(new_timestamp - self.last_timestamp)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        self.last_timestamp = new_timestamp;
        self.instant_fps = 1.0 / dt;
    }
};

const Detector = struct {
    const Self = @This();

    model: Yolo,

    // temporary buffer for the cropped image that feeds into the model.
    // TODO: optimization: instead of filling a cropped version every frame, feed the infer
    // function with the uncropped version and let it fill its internal f32 buffer with the
    // cropped version directly. this is currently just an intermediary buffer.
    crop_buffer: []u8,

    inference_running: bool = false, // have we submitted a frame?
    current_center: [2]f32 = .{ HALF_W, HALF_H }, //crop center of the most-recently submitted frame

    last_pred: Yolo.Prediction = .{}, // results of the last successful inference
    last_pred_center: [2]f32 = .{ HALF_W, HALF_H }, // crop center of the last successful inference

    target_detected: bool = false, // is there a target in the last prediction?
    target_center: [2]f32 = .{ HALF_W, HALF_H },

    pub fn init(arena: std.mem.Allocator) !Self {
        const crop_buffer = try arena.alloc(u8, Yolo.WIDTH * Yolo.HEIGHT * 3);
        @memset(crop_buffer, 0); // Initialize to black

        return .{
            .model = try Yolo.init(arena),
            .crop_buffer = crop_buffer,
        };
    }

    pub fn deinit(self: *Self) void {
        self.model.deinit();
    }

    pub fn submitFrame(self: *Self, frame: []const u8, crop_center: [2]f32) !void {
        if (self.inference_running) return;

        try pixels.cropRgbFrame(
            frame,
            WIDTH,
            HEIGHT,
            self.crop_buffer,
            Yolo.WIDTH,
            Yolo.HEIGHT,
            @intFromFloat(crop_center[0] - HALF_YOLO_W),
            @intFromFloat(crop_center[1] - HALF_YOLO_H),
        );

        try self.model.startInfer(self.crop_buffer, Yolo.WIDTH, Yolo.HEIGHT);
        self.inference_running = true;
    }

    pub fn collectResult(self: *Self) !bool {
        if (!self.inference_running) return false;
        const pred = (try self.model.pollResult()) orelse return false;

        self.last_pred = pred;
        self.last_pred_center = self.current_center;
        self.inference_running = false;

        self.target_detected = pred.score > CONFIDENCE_THRESHOLD;
        if (self.target_detected) {
            self.target_center = .{
                self.last_pred_center[0] - HALF_YOLO_W + pred.cx,
                self.last_pred_center[1] - HALF_YOLO_H + pred.cy,
            };
        }

        return true;
    }
};

fn drawFpsCounter(fps: f32) void {
    var fps_text_buffer: [64]u8 = undefined;
    const fps_text = std.fmt.bufPrintZ(&fps_text_buffer, "FPS: {d:.2}", .{fps}) catch unreachable;
    rl.drawText(fps_text, 10, 10, 20, rl.Color.lime);
}

fn drawBoundingBox(pred: Yolo.Prediction, result_center: [2]f32) void {
    // convert yolo [cx, cy, w, h] into raylib [x, y, w, h]
    const box_x = pred.cx - (pred.w / 2.0);
    const box_y = pred.cy - (pred.h / 2.0);

    const rect = rl.Rectangle{
        .x = box_x + result_center[0] - HALF_YOLO_W,
        .y = box_y + result_center[1] - HALF_YOLO_H,
        .width = pred.w,
        .height = pred.h,
    };

    rl.drawRectangleLinesEx(rect, 3.0, rl.Color.lime);

    var label_buffer: [64]u8 = undefined;
    const label_text = std.fmt.bufPrintZ(&label_buffer, "{s}: {d:.2}%", .{
        Yolo.model_labels[pred.class_id],
        pred.score * 100.0,
    }) catch unreachable;

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
