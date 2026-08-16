const std = @import("std");

pub const FRAME_WIDTH = 1280;
pub const FRAME_HEIGHT = 720;
pub const FRAME_BYTES = FRAME_WIDTH * FRAME_HEIGHT * 3; // rgb888

pub const CROP_WIDTH = 640;
pub const CROP_HEIGHT = 640;
