const std = @import("std");
const ort = @import("ort");

const model_data: []const u8 = @embedFile("yolov8m.onnx");
// in 0: name='images', shape={ 1, 3, 640, 640 }
// out 0: name='output0', shape={ 1, 84, 8400 }

pub const model_labels = block: {
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

const YOLO_WIDTH = 640;
const YOLO_HEIGHT = 640;

const Self = @This();

api: *const ort.OrtApi,
env: *ort.OrtEnv,
session: *ort.OrtSession,
allocator: *ort.OrtAllocator,
memory_info: *ort.OrtMemoryInfo,
input_tensor_data: []f32,
input_shape: [4]i64,

pub fn init(arena: std.mem.Allocator) !Self {
    const api_base = ort.OrtGetApiBase().?;
    const api = api_base.*.GetApi.?(ort.ORT_API_VERSION);

    var env: ?*ort.OrtEnv = null;
    try checkStatus(api, api.*.CreateEnv.?(ort.ORT_LOGGING_LEVEL_WARNING, "YOLO", &env));
    errdefer api.*.ReleaseEnv.?(env);

    var session_options: ?*ort.OrtSessionOptions = null;
    try checkStatus(api, api.*.CreateSessionOptions.?(&session_options));
    defer api.*.ReleaseSessionOptions.?(session_options);

    // prevent cpu from drawing 200 watts. it draw 140 now :(.
    try checkStatus(api, api.*.SetSessionGraphOptimizationLevel.?(session_options, ort.ORT_ENABLE_ALL));
    // don't hammer all the cores.
    try checkStatus(api, api.*.SetIntraOpNumThreads.?(session_options, 4));

    // enable cuda so we can still use the computer
    var cuda_options: ?*ort.OrtCUDAProviderOptionsV2 = null;
    try checkStatus(api, api.*.CreateCUDAProviderOptions.?(&cuda_options));
    defer api.*.ReleaseCUDAProviderOptions.?(cuda_options);

    try checkStatus(api, api.*.SessionOptionsAppendExecutionProvider_CUDA_V2.?(session_options, cuda_options));

    // create session from the model data
    var session: ?*ort.OrtSession = null;
    try checkStatus(api, api.*.CreateSessionFromArray.?(
        env,
        model_data.ptr,
        model_data.len,
        session_options,
        &session,
    ));
    errdefer api.*.ReleaseSession.?(session);

    var allocator: ?*ort.OrtAllocator = null;
    try checkStatus(api, api.*.GetAllocatorWithDefaultOptions.?(&allocator));

    // setup memory and buffers

    var memory_info: ?*ort.OrtMemoryInfo = null;
    try checkStatus(api, api.*.CreateCpuMemoryInfo.?(
        ort.OrtArenaAllocator,
        ort.OrtMemTypeDefault,
        &memory_info,
    ));
    errdefer api.*.ReleaseMemoryInfo.?(memory_info);

    // arena allocate buffer for the input tensor. we already know the size from the previous debug prints.
    const input_tensor_len = 1 * 3 * YOLO_HEIGHT * YOLO_WIDTH;
    const input_tensor_data = try arena.alloc(f32, input_tensor_len);
    const input_shape = [_]i64{ 1, 3, YOLO_HEIGHT, YOLO_WIDTH };

    return .{
        .api = api.?,
        .env = env.?,
        .session = session.?,
        .allocator = allocator.?,
        .memory_info = memory_info.?,
        .input_tensor_data = input_tensor_data,
        .input_shape = input_shape,
    };
}

pub fn deinit(self: *Self) void {
    self.api.*.ReleaseMemoryInfo.?(self.memory_info);
    self.api.*.ReleaseSession.?(self.session);
    self.api.*.ReleaseEnv.?(self.env);
}

pub fn infer(self: *Self, rgb_frame: []const u8, src_w: usize, src_h: usize) !Prediction {
    preprocessYolo(
        rgb_frame,
        self.input_tensor_data,
        src_w,
        src_h,
        YOLO_WIDTH,
        YOLO_HEIGHT,
    );

    // FIXME: the documentation says we need to free these with the allocator. how do we do that?
    var input_name: ?[*]u8 = null;
    try checkStatus(self.api, self.api.*.SessionGetInputName.?(self.session, 0, self.allocator, &input_name));

    var output_name: ?[*]u8 = null;
    try checkStatus(self.api, self.api.*.SessionGetOutputName.?(self.session, 0, self.allocator, &output_name));

    // TODO: do we need to create this tensor every time? can we do it once outside the loop and fill the data every frame? experiment...
    var input_tensor: ?*ort.OrtValue = null;
    try checkStatus(self.api, self.api.*.CreateTensorWithDataAsOrtValue.?(
        self.memory_info,
        @ptrCast(self.input_tensor_data.ptr),
        self.input_tensor_data.len * @sizeOf(f32),
        @ptrCast(&self.input_shape[0]),
        self.input_shape.len,
        ort.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor,
    ));
    defer self.api.*.ReleaseValue.?(input_tensor);

    var output_tensor: ?*ort.OrtValue = null;
    try checkStatus(self.api, self.api.*.Run.?(
        self.session,
        null, // run_options
        &input_name,
        &input_tensor,
        1,
        &output_name,
        1,
        &output_tensor,
    ));
    defer self.api.*.ReleaseValue.?(output_tensor);

    var out_ptr: [*c]f32 = null;
    try checkStatus(self.api, self.api.*.GetTensorMutableData.?(output_tensor, @ptrCast(&out_ptr)));

    const top = getTopPrediction(out_ptr, 8400);
    return top;
}

pub const Prediction = struct {
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

    const r_offset = 0;
    const g_offset = channel_stride;
    const b_offset = channel_stride * 2;

    const scale: f32 = 1.0 / 255.0;

    var src_row_start: usize = 0;
    var dst_row_start: usize = 0;

    for (0..max_y) |_| {
        var src_idx = src_row_start;
        var dst_idx = dst_row_start;

        for (0..max_x) |_| {
            tensor[r_offset + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 0])) * scale;
            tensor[g_offset + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 1])) * scale;
            tensor[b_offset + dst_idx] = @as(f32, @floatFromInt(rgb[src_idx + 2])) * scale;

            src_idx += 3;
            dst_idx += 1;
        }

        src_row_start += src_w * 3;
        dst_row_start += dst_w;
    }
}

// Copies the source RGB array into the destination array, cropping the frame to fit.
// The caller must ensure that the destination is memset to 0.
pub fn cropRgbFrame(src_rgb: []const u8, src_w: usize, src_h: usize, dst_rgb: []u8, dst_w: usize, dst_h: usize, x0: usize, y0: usize) !void {
    if (src_rgb.len != src_w * src_h * 3) return error.InvalidBufferSize;
    if (dst_rgb.len != dst_w * dst_h * 3) return error.InvalidBufferSize;

    // completely outside the src image.
    if (x0 >= src_w or y0 >= src_h) return;

    const copy_w = @min(dst_w, src_w - x0);
    const copy_h = @min(dst_h, src_h - y0);

    for (0..copy_h) |y| {
        const src_y = y0 + y;

        const src_idx = ((src_w * src_y) + x0) * 3;
        const dst_idx = (dst_w * y) * 3;

        @memcpy(dst_rgb[dst_idx .. dst_idx + copy_w * 3], src_rgb[src_idx .. src_idx + copy_w * 3]);
    }
}

fn checkStatus(api: *const ort.OrtApi, status: ?*ort.OrtStatus) !void {
    if (status) |st| {
        const msg = api.*.GetErrorMessage.?(st);
        std.debug.print("ort error: {s}\n", .{msg});
        api.*.ReleaseStatus.?(st);
        return error.OrtError;
    }
}
