const std = @import("std");
const onnx = @import("onnx");
const vl = @import("vl");

const DEVICE = "/dev/video0";
const WIDTH = 640;
const HEIGHT = 480;
const NUM_BUFFERS = 4;
const TIMEOUT = 2000;

fn ioctl(fd: std.os.linux.fd_t, request: u32, arg: usize) !void {
    var r = std.os.linux.ioctl(fd, request, arg);
    while (std.os.linux.errno(r) == .INTR) r = std.os.linux.ioctl(fd, request, arg);
    const errno = std.os.linux.errno(r);
    if (errno != .SUCCESS) return error.FailedIoctl;
}

const Buffer = struct {
    start: []align(std.heap.page_size_min) u8,
    length: usize,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    _ = io;
    _ = arena;

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

    while (true) {
        _ = try std.posix.poll(&pfds, TIMEOUT);

        if ((pfds[0].revents & std.posix.POLL.IN) != 0) {
            buf = vl.v4l2_buffer{};
            buf.type = vl.V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = vl.V4L2_MEMORY_MMAP;
            try ioctl(fd, vl.VIDIOC_DQBUF, @intFromPtr(&buf));

            try ioctl(fd, vl.VIDIOC_QBUF, @intFromPtr(&buf));
        }
    }
}
