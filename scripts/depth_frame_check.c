/* Bounded Linux V4L2 Z16 depth capture, using only libc and kernel headers.
 * Build: gcc -std=c11 -O2 -Wall -Wextra -Wpedantic depth_frame_check.c -o depth_frame_check
 * Run: timeout --kill-after=3 20 ./depth_frame_check --device /dev/video0 \
 *          --seconds 10 --output /tmp/depth-last.pgm
 *
 * This opens only the explicitly named capture node and temporarily changes its
 * format. It does not change controls, frame interval, firmware, or services.
 * The original format is restored after buffers are released. Do not run on a
 * node owned by another application. Kernel ioctl stalls require an external
 * timeout. PGM values are raw depth units, NOT verified millimetres; this tool
 * does not establish metric accuracy or the health of the separate RGB node.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <linux/videodev2.h>

#ifndef V4L2_PIX_FMT_Z16
#define V4L2_PIX_FMT_Z16 v4l2_fourcc('Z', '1', '6', ' ')
#endif

#define WIDTH 640U
#define HEIGHT 480U
#define MAX_BUFFERS 8U
#define MAX_SEQUENCES 65536U

static volatile sig_atomic_t interrupted;

static void on_signal(int number) { interrupted = number; }

static int xioctl(int fd, unsigned long request, void *arg) {
    int result;
    do { result = ioctl(fd, request, arg); } while (result < 0 && errno == EINTR);
    return result;
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) < 0) {
        perror("clock_gettime");
        return -1.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void print_format(const char *label, const struct v4l2_format *fmt) {
    const struct v4l2_pix_format *p = &fmt->fmt.pix;
    printf("%s width=%u height=%u fourcc=%c%c%c%c bytesperline=%u sizeimage=%u field=%u\n",
           label, p->width, p->height, p->pixelformat & 255,
           (p->pixelformat >> 8) & 255, (p->pixelformat >> 16) & 255,
           (p->pixelformat >> 24) & 255, p->bytesperline, p->sizeimage, p->field);
}

static bool same_format(const struct v4l2_format *a, const struct v4l2_format *b) {
    return a->fmt.pix.width == b->fmt.pix.width &&
           a->fmt.pix.height == b->fmt.pix.height &&
           a->fmt.pix.pixelformat == b->fmt.pix.pixelformat &&
           a->fmt.pix.field == b->fmt.pix.field &&
           a->fmt.pix.bytesperline == b->fmt.pix.bytesperline &&
           a->fmt.pix.sizeimage == b->fmt.pix.sizeimage;
}

static int write_pgm(const char *path, const uint8_t *frame) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) { perror("create PGM (will not overwrite)"); return -1; }
    FILE *file = fdopen(fd, "wb");
    if (!file) { perror("fdopen PGM"); close(fd); unlink(path); return -1; }
    int failed = fprintf(file, "P5\n# Raw Z16 values; depth scale not verified\n%u %u\n65535\n",
                         WIDTH, HEIGHT) < 0;
    uint8_t row[WIDTH * 2];
    for (unsigned y = 0; y < HEIGHT && !failed; y++) {
        for (unsigned x = 0; x < WIDTH; x++) {
            size_t offset = ((size_t)y * WIDTH + x) * 2;
            row[x * 2] = frame[offset + 1];
            row[x * 2 + 1] = frame[offset];
        }
        failed = fwrite(row, 1, sizeof(row), file) != sizeof(row);
    }
    if (fclose(file) != 0) failed = 1;
    if (failed) { fprintf(stderr, "PGM write failed\n"); unlink(path); return -1; }
    printf("output_pgm=%s\n", path);
    return 0;
}

struct mapping { void *data; size_t length; };

int main(int argc, char **argv) {
    const char *device = NULL, *output = NULL;
    long seconds = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc && !device) {
            device = argv[++i];
        } else if (!strcmp(argv[i], "--output") && i + 1 < argc && !output) {
            output = argv[++i];
        } else if (!strcmp(argv[i], "--seconds") && i + 1 < argc && !seconds) {
            char *end;
            errno = 0;
            seconds = strtol(argv[++i], &end, 10);
            if (errno || *end || seconds < 1 || seconds > 30) seconds = -1;
        } else {
            fprintf(stderr, "Unexpected, repeated, or incomplete argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (!device || strncmp(device, "/dev/video", 10) || !device[10] ||
        strspn(device + 10, "0123456789") != strlen(device + 10) ||
        !output || !*output || seconds < 1 || seconds > 30) {
        fprintf(stderr, "Usage: %s --device /dev/videoN --seconds 1..30 --output NEW.pgm\n", argv[0]);
        return 2;
    }
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL)) {
        perror("sigaction"); return 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    int fd = -1, failed = 0;
    bool format_changed = false, buffers_requested = false, streaming = false;
    struct mapping maps[MAX_BUFFERS] = {{0}};
    struct v4l2_format original = {0}, format = {0};
    uint8_t *last_frame = NULL;
    uint32_t *sequences = NULL;
    unsigned mapped = 0, unique = 0;
    uint64_t dequeued = 0, complete = 0, error_frames = 0, short_frames = 0;
    uint64_t all_zero = 0, duplicates = 0, missing = 0, backwards = 0;
    uint64_t repeated_times = 0, backwards_times = 0, invalid_times = 0;
    uint64_t pixels = 0, nonzero = 0, last_nonzero = 0;
    unsigned minimum = UINT16_MAX, maximum = 0, last_minimum = 0, last_maximum = 0;
    uint32_t previous_seq = 0, first_seq = 0, last_seq = 0;
    int64_t first_time = 0, previous_time = 0, last_time = 0;
    uint32_t timestamp_flags = 0;
    double started = 0, elapsed = 0;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

#define FAIL(message) do { perror(message); failed = 1; goto cleanup; } while (0)
#define REJECT(message) do { fprintf(stderr, "%s\n", message); failed = 1; goto cleanup; } while (0)

    fd = open(device, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) FAIL("open capture node");
    struct stat st;
    if (fstat(fd, &st) < 0) FAIL("fstat capture node");
    if (!S_ISCHR(st.st_mode)) REJECT("Capture node is not a character device");
    struct v4l2_capability caps = {0};
    if (xioctl(fd, VIDIOC_QUERYCAP, &caps) < 0) FAIL("VIDIOC_QUERYCAP");
    uint32_t capabilities = caps.capabilities & V4L2_CAP_DEVICE_CAPS ?
                            caps.device_caps : caps.capabilities;
    printf("device=%s driver=%.16s card=%.32s bus=%.32s capabilities=0x%08x\n",
           device, caps.driver, caps.card, caps.bus_info, capabilities);
    if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(capabilities & V4L2_CAP_STREAMING))
        REJECT("Requires single-plane VIDEO_CAPTURE and STREAMING capabilities");
    bool z16 = false;
    for (unsigned index = 0; index < 256; index++) {
        struct v4l2_fmtdesc desc = {0};
        desc.type = type;
        desc.index = index;
        if (xioctl(fd, VIDIOC_ENUM_FMT, &desc) < 0) {
            if (errno == EINVAL) break;
            FAIL("VIDIOC_ENUM_FMT");
        }
        if (desc.pixelformat == V4L2_PIX_FMT_Z16) z16 = true;
    }
    if (!z16) REJECT("Selected capture node does not advertise Z16 depth format");
    original.type = type;
    if (xioctl(fd, VIDIOC_G_FMT, &original) < 0) FAIL("VIDIOC_G_FMT original");
    print_format("original_format", &original);
    format = original;
    format.fmt.pix.width = WIDTH;
    format.fmt.pix.height = HEIGHT;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_Z16;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    format.fmt.pix.bytesperline = 0;
    format.fmt.pix.sizeimage = 0;
    if (interrupted) REJECT("Interrupted before format configuration");
    if (xioctl(fd, VIDIOC_S_FMT, &format) < 0) FAIL("VIDIOC_S_FMT (node may be busy)");
    format_changed = true;
    if (xioctl(fd, VIDIOC_G_FMT, &format) < 0) FAIL("VIDIOC_G_FMT configured");
    print_format("capture_format", &format);
    if (format.fmt.pix.width != WIDTH || format.fmt.pix.height != HEIGHT ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_Z16 ||
        format.fmt.pix.field != V4L2_FIELD_NONE ||
        format.fmt.pix.bytesperline < WIDTH * 2 ||
        format.fmt.pix.bytesperline > WIDTH * 2 + 65536U)
        REJECT("Driver did not provide requested 640x480 progressive Z16 format with sane stride");
    size_t required = (size_t)(HEIGHT - 1) * format.fmt.pix.bytesperline + WIDTH * 2;
    if (format.fmt.pix.sizeimage < required)
        REJECT("Driver sizeimage is too small for the configured Z16 frame");
    last_frame = malloc((size_t)WIDTH * HEIGHT * 2);
    sequences = malloc((size_t)MAX_SEQUENCES * sizeof(*sequences));
    if (!last_frame || !sequences) FAIL("malloc capture storage");
    struct v4l2_requestbuffers request = {0};
    request.count = 4;
    request.type = type;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &request) < 0) FAIL("VIDIOC_REQBUFS");
    buffers_requested = true;
    if (!request.count || request.count > MAX_BUFFERS) REJECT("Unexpected capture buffer count");
    for (unsigned i = 0; i < request.count; i++) {
        struct v4l2_buffer buffer = {0};
        buffer.type = type;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = i;
        if (xioctl(fd, VIDIOC_QUERYBUF, &buffer) < 0) FAIL("VIDIOC_QUERYBUF");
        if (buffer.length < required || buffer.length > 64U * 1024U * 1024U)
            REJECT("Capture buffer length is invalid");
        maps[i].length = buffer.length;
        maps[i].data = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, buffer.m.offset);
        if (maps[i].data == MAP_FAILED) { maps[i].data = NULL; FAIL("mmap"); }
        mapped++;
        if (xioctl(fd, VIDIOC_QBUF, &buffer) < 0) FAIL("VIDIOC_QBUF initial");
    }
    if (interrupted) REJECT("Interrupted before streaming");
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) FAIL("VIDIOC_STREAMON");
    streaming = true;
    started = now_seconds();
    if (started < 0) REJECT("Cannot establish monotonic capture deadline");
    for (;;) {
        if (interrupted) break;
        double current = now_seconds();
        if (current < 0) REJECT("Cannot read capture deadline clock");
        double remaining = started + seconds - current;
        if (remaining <= 0) break;
        int timeout_ms = remaining < 0.25 ? (int)(remaining * 1000) + 1 : 250;
        struct pollfd poll_fd = {.fd = fd, .events = POLLIN | POLLPRI};
        int ready = poll(&poll_fd, 1, timeout_ms);
        if (ready < 0) { if (errno == EINTR) continue; FAIL("poll capture"); }
        if (!ready) continue;
        if (poll_fd.revents & (POLLERR | POLLHUP | POLLNVAL))
            REJECT("Capture poll reported error, hangup, or invalid fd");
        if (!(poll_fd.revents & (POLLIN | POLLPRI))) continue;
        struct v4l2_buffer buffer = {0};
        buffer.type = type;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) continue;
            FAIL("VIDIOC_DQBUF");
        }
        dequeued++;
        if (buffer.index >= mapped) REJECT("Driver returned an invalid buffer index");
        if (buffer.bytesused > maps[buffer.index].length)
            REJECT("Driver bytesused exceeds mapped buffer length");
        bool seen = false;
        for (unsigned i = 0; i < unique; i++) {
            if (sequences[i] == buffer.sequence) { seen = true; break; }
        }
        if (seen) duplicates++;
        else {
            if (unique == MAX_SEQUENCES) REJECT("Sequence tracking limit exceeded");
            sequences[unique++] = buffer.sequence;
        }
        int64_t stamp = (int64_t)buffer.timestamp.tv_sec * 1000000 + buffer.timestamp.tv_usec;
        if (buffer.timestamp.tv_sec < 0 || buffer.timestamp.tv_usec < 0 ||
            buffer.timestamp.tv_usec >= 1000000 || stamp <= 0) invalid_times++;
        if (dequeued == 1) {
            first_seq = buffer.sequence;
            first_time = stamp;
            timestamp_flags = buffer.flags & (V4L2_BUF_FLAG_TIMESTAMP_MASK | V4L2_BUF_FLAG_TSTAMP_SRC_MASK);
        } else {
            uint32_t delta = buffer.sequence - previous_seq;
            if (delta > 1 && delta < UINT32_C(0x80000000)) missing += delta - 1;
            else if (delta >= UINT32_C(0x80000000)) backwards++;
            if (stamp == previous_time) repeated_times++;
            else if (stamp < previous_time) backwards_times++;
        }
        previous_seq = last_seq = buffer.sequence;
        previous_time = last_time = stamp;
        if (buffer.flags & V4L2_BUF_FLAG_ERROR) error_frames++;
        if (buffer.bytesused < required) short_frames++;
        if (!(buffer.flags & V4L2_BUF_FLAG_ERROR) && buffer.bytesused >= required) {
            uint64_t count = 0;
            unsigned low = UINT16_MAX, high = 0;
            for (unsigned y = 0; y < HEIGHT; y++) {
                const uint8_t *row = (const uint8_t *)maps[buffer.index].data +
                                     (size_t)y * format.fmt.pix.bytesperline;
                memcpy(last_frame + (size_t)y * WIDTH * 2, row, WIDTH * 2);
                for (unsigned x = 0; x < WIDTH; x++) {
                    unsigned value = row[x * 2] | (unsigned)row[x * 2 + 1] << 8;
                    if (!value) continue;
                    count++;
                    if (value < low) low = value;
                    if (value > high) high = value;
                }
            }
            complete++;
            if (!count) all_zero++;
            pixels += (uint64_t)WIDTH * HEIGHT;
            nonzero += count;
            if (count && low < minimum) minimum = low;
            if (high > maximum) maximum = high;
            last_nonzero = count;
            last_minimum = count ? low : 0;
            last_maximum = high;
        }
        if (xioctl(fd, VIDIOC_QBUF, &buffer) < 0) FAIL("VIDIOC_QBUF recycle");
    }

cleanup:
    if (started > 0) {
        double stopped = now_seconds();
        if (stopped >= started) elapsed = stopped - started;
        else failed = 1;
    }
    if (streaming) {
        if (xioctl(fd, VIDIOC_STREAMOFF, &type) < 0) { perror("VIDIOC_STREAMOFF"); failed = 1; }
        else printf("streamoff=ok\n");
    }
    for (unsigned i = 0; i < mapped; i++) {
        if (munmap(maps[i].data, maps[i].length) < 0) { perror("munmap"); failed = 1; }
    }
    if (buffers_requested) {
        struct v4l2_requestbuffers release = {0};
        release.type = type;
        release.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_REQBUFS, &release) < 0) { perror("VIDIOC_REQBUFS release"); failed = 1; }
    }
    if (format_changed) {
        struct v4l2_format restored = original;
        if (xioctl(fd, VIDIOC_S_FMT, &restored) < 0 ||
            xioctl(fd, VIDIOC_G_FMT, &restored) < 0) {
            perror("restore original format"); failed = 1;
        } else if (!same_format(&restored, &original)) {
            fprintf(stderr, "Original format did not restore exactly\n"); failed = 1;
        } else printf("format_restore=ok\n");
    }
    if (fd >= 0 && close(fd) < 0) { perror("close capture node"); failed = 1; }
    if (complete && write_pgm(output, last_frame) < 0) failed = 1;
    if (interrupted || complete < 2 || !nonzero || all_zero || error_frames || short_frames ||
        duplicates || missing || backwards || repeated_times || backwards_times || invalid_times)
        failed = 1;
    printf("summary elapsed_seconds=%.6f requested_seconds=%ld dequeued=%" PRIu64
           " complete_frames=%" PRIu64 " observed_fps=%.3f unique_sequences=%u"
           " duplicate_sequences=%" PRIu64 " missing_sequences=%" PRIu64
           " backwards_sequences=%" PRIu64 " error_flag_frames=%" PRIu64
           " short_frames=%" PRIu64 " all_zero_frames=%" PRIu64 "\n",
           elapsed, seconds, dequeued, complete, elapsed > 0 ? complete / elapsed : 0,
           unique, duplicates, missing, backwards, error_frames, short_frames, all_zero);
    printf("timestamps first_sequence=%u last_sequence=%u first_us=%" PRId64
           " last_us=%" PRId64 " first_flags=0x%08x repeated=%" PRIu64
           " backwards=%" PRIu64 " invalid=%" PRIu64 "\n",
           first_seq, last_seq, first_time, last_time, timestamp_flags,
           repeated_times, backwards_times, invalid_times);
    printf("depth raw_units=true total_pixels=%" PRIu64 " nonzero_pixels=%" PRIu64
           " nonzero_fraction=%.6f nonzero_min=%u max=%u last_nonzero_fraction=%.6f"
           " last_nonzero_min=%u last_max=%u\n",
           pixels, nonzero, pixels ? (double)nonzero / pixels : 0,
           nonzero ? minimum : 0, maximum, (double)last_nonzero / ((uint64_t)WIDTH * HEIGHT),
           last_minimum, last_maximum);
    printf("result=%s interrupted_signal=%d scope=bounded_raw_depth_stream_only\n",
           failed ? "incomplete_or_anomalous" : "frames_received_without_observed_stream_errors",
           (int)interrupted);
    free(last_frame);
    free(sequences);
    return interrupted ? 128 + interrupted : failed;
}
