const std = @import("std");

pub const FRAME_WIDTH = 1280;
pub const FRAME_HEIGHT = 720;
pub const FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3; // rgb888

pub const CROP_WIDTH = 640;
pub const CROP_HEIGHT = 640;

const CROP_W_F32: f32 = @floatFromInt(CROP_WIDTH);
const CROP_H_F32: f32 = @floatFromInt(CROP_HEIGHT);
const HALF_CROP_W: f32 = CROP_W_F32 / 2.0;
const HALF_CROP_H: f32 = CROP_H_F32 / 2.0;

const FRAME_W_F32: f32 = @floatFromInt(FRAME_WIDTH);
const FRAME_H_F32: f32 = @floatFromInt(FRAME_HEIGHT);
const HALF_FRAME_W: f32 = FRAME_W_F32 / 2.0;
const HALF_FRAME_H: f32 = FRAME_H_F32 / 2.0;

pub const VIDEO_PORT = 9500;
pub const META_PORT = 9501;

pub const FrameHeader = packed struct {
    len: u64, // length of the jpeg
    timestamp_ns: i96, // frame generation time

    crop_center_x: f32 = HALF_FRAME_W,
    crop_center_y: f32 = HALF_FRAME_H,
};
