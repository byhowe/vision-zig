#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <poll.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

// toy implementation to see how it works.

#define DEVICE "/dev/video0"
#define WIDTH 640
#define HEIGHT 480
#define NUM_BUFFERS 4

struct Buffer {
  void *start;
  size_t length;
};

static int xioctl(int fd, unsigned long request, void *arg) {
  int r;
  do {
    r = ioctl(fd, request, arg);
  } while (-1 == r && errno == EINTR); // retry on interrupt
  return r;
}

int main() {
  int fd = open(DEVICE, O_RDWR);
  if (fd < 0) {
    perror("opening video device failed");
    return 1;
  }

  // configure the camera
  struct v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = WIDTH;
  fmt.fmt.pix.height = HEIGHT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;

  if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    perror("setting pixel format failed");
    close(fd);
    return 1;
  }

  // request a buffer
  struct v4l2_requestbuffers req = {0};
  req.count = NUM_BUFFERS;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
    perror("requesting buffers failed");
    close(fd);
    return 1;
  }

  struct Buffer buffers[NUM_BUFFERS];

  for (int i = 0; i < NUM_BUFFERS; i += 1) {
    // query the requested buffer's memory
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
      perror("querying buffer failed");
      close(fd);
      return 1;
    }

    // map the kernel memory into user space
    buffers[i].length = buf.length;
    buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buf.m.offset);

    if (buffers[i].start == MAP_FAILED) {
      perror("mmap failed");
      close(fd);
      return 1;
    }

    // queue the empty buffer so that the camera driver can fill it
    if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
      perror("queue buffer failed");
      close(fd);
      return 1;
    }
  }

  // start streaming
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
    perror("start stream failed");
    close(fd);
    return 1;
  }

  printf("camera initialized successfully. capturing...\n");

  struct pollfd pfd = {0};
  pfd.fd = fd;
  pfd.events = POLLIN;

  for (int frame_count = 0; frame_count < 10; frame_count += 1) {
    int result = poll(&pfd, 1, 2000);

    if (result < 0) {
      perror("poll error");
      break;
    } else if (result == 0) {
      printf("timeout. waiting for frame...\n");
      continue;
    }

    // kernel driver captured a frame
    if (pfd.revents & POLLIN) {
      struct v4l2_buffer buf = {0};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      // dequeue the buffer that the kernel filled
      if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
        perror("dequeue buffer failed");
        break;
      }

      printf("[frame %d] ready frame. buffer index: %d, bytes: %u\n", frame_count + 1, buf.index, buf.bytesused);

      // re-queue buffer so that the kernel can use it
      if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
        perror("queue buffer failed");
        break;
      }
    }
  }

  // stop stream
  if (xioctl(fd, VIDIOC_STREAMOFF, &type) < 0) {
    perror("stop stream failed");
    close(fd);
    return 1;
  }

  for (int i = 0; i < NUM_BUFFERS; i += 1) {
    munmap(buffers[i].start, buffers[i].length);
  }

  close(fd);

  return 0;
}
