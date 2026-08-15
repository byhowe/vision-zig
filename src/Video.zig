const std = @import("std");
const vl = @import("vl");

const NUM_BUFFERS = 4; // sensible default

const Self = @This();

fd: std.posix.fd_t,
buffers: []Buffer = undefined,

const Buffer = []align(std.heap.page_size_min) u8;

pub fn init(path: []const u8) !Self {
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        path,
        .{ .ACCMODE = .RDWR, .NONBLOCK = true, .CLOEXEC = true },
        0,
    );
    errdefer std.Io.Threaded.closeFd(fd);

    // query capabilities to make sure cam supports video capture and streaming
    var cap = std.mem.zeroes(vl.v4l2_capability);
    try ioctl(fd, vl.VIDIOC_QUERYCAP, @intFromPtr(&cap));

    if (0 == (cap.capabilities & vl.V4L2_CAP_VIDEO_CAPTURE)) return error.CapVideoCapture;
    if (0 == (cap.capabilities & vl.V4L2_CAP_STREAMING)) return error.CapStreaming;

    return .{
        .fd = fd,
    };
}

pub fn deinit(self: *Self) void {
    std.Io.Threaded.closeFd(self.fd);
}

pub fn setFormat(self: *Self, width: usize, height: usize) !void {
    var fmt = vl.v4l2_format{};
    fmt.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = @intCast(width);
    fmt.fmt.pix.height = @intCast(height);
    // FIXME: currently hard coding the pixel format. take it from the user.
    fmt.fmt.pix.pixelformat = vl.V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = vl.V4L2_FIELD_NONE;
    try ioctl(self.fd, vl.VIDIOC_S_FMT, @intFromPtr(&fmt));
}

pub fn setFramerate(self: *Self, framerate: usize) !usize {
    var parm = vl.v4l2_streamparm{};
    parm.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = @intCast(framerate);
    try ioctl(self.fd, vl.VIDIOC_S_PARM, @intFromPtr(&parm));

    // return the actual fps. the driver may have set it to something else.
    return @intCast(parm.parm.capture.timeperframe.denominator);
}

pub fn requestBuffers(self: *Self, arena: std.mem.Allocator) !void {
    var req = vl.v4l2_requestbuffers{};
    req.count = NUM_BUFFERS;
    req.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = vl.V4L2_MEMORY_MMAP;
    try ioctl(self.fd, vl.VIDIOC_REQBUFS, @intFromPtr(&req));

    // set the internal number of buffers
    self.buffers = try arena.alloc(Buffer, @intCast(req.count));
}

pub fn mapBuffers(self: *Self) !void {
    for (0..self.buffers.len) |i| {
        var buffer = vl.v4l2_buffer{};
        buffer.index = @intCast(i);
        buffer.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = vl.V4L2_MEMORY_MMAP;
        try ioctl(self.fd, vl.VIDIOC_QUERYBUF, @intFromPtr(&buffer));

        const mem = try std.posix.mmap(
            null,
            buffer.length,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            self.fd,
            buffer.m.offset,
        );

        self.buffers[i] = mem;
    }
}

pub fn unmapBuffers(self: *Self) !void {
    for (0..self.buffers.len) |i| std.posix.munmap(self.buffers[i]);
}

pub fn queueBuffers(self: *Self) !void {
    for (0..self.buffers.len) |i| try self.queueBuffer(i);
}

pub fn queueBuffer(self: *Self, idx: usize) !void {
    var buffer = std.mem.zeroes(vl.v4l2_buffer);
    buffer.index = @intCast(idx);
    buffer.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = vl.V4L2_MEMORY_MMAP;
    try ioctl(self.fd, vl.VIDIOC_QBUF, @intFromPtr(&buffer));
}

pub fn dequeueBuffer(self: *Self) !usize {
    var buf = std.mem.zeroes(vl.v4l2_buffer);
    buf.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = vl.V4L2_MEMORY_MMAP;
    try ioctl(self.fd, vl.VIDIOC_DQBUF, @intFromPtr(&buf));

    return buf.index;
}

pub fn streamon(self: *Self) !void {
    const ty = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    try ioctl(self.fd, vl.VIDIOC_STREAMON, @intFromPtr(&ty));
}

pub fn streamoff(self: *Self) !void {
    const ty = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
    try ioctl(self.fd, vl.VIDIOC_STREAMOFF, @intFromPtr(&ty));
}

fn ioctl(fd: std.posix.fd_t, request: u32, arg: usize) !void {
    while (true) {
        const rc = std.os.linux.ioctl(fd, request, arg);
        switch (std.os.linux.errno(rc)) {
            .SUCCESS => return,
            .INTR => continue,
            else => |err| {
                std.log.err("ioctl 0x{X} failed: {s}", .{ request, @tagName(err) });
                return error.IoctlFailed;
            },
        }
    }
}
