import cv2
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
device = cuda.Device(0)


BLOCK_SIZE = 24

cuda_kernel = """
__global__ void pixelar(unsigned char* img, int w, int h, int step, int block_size) {
    int bx = blockIdx.x * block_size;
    int by = blockIdx.y * block_size;

    if (bx >= w || by >= h) return;

    int r_sum = 0, g_sum = 0, b_sum = 0;
    int count = 0;

    for (int dy = 0; dy < block_size && (by + dy) < h; dy++) {
        for (int dx = 0; dx < block_size && (bx + dx) < w; dx++) {
            int idx = (by + dy) * step + (bx + dx) * 3;
            b_sum += img[idx];
            g_sum += img[idx + 1];
            r_sum += img[idx + 2];
            count++;
        }
    }

    unsigned char r_avg = r_sum / count;
    unsigned char g_avg = g_sum / count;
    unsigned char b_avg = b_sum / count;

    for (int dy = 0; dy < block_size && (by + dy) < h; dy++) {
        for (int dx = 0; dx < block_size && (bx + dx) < w; dx++) {
            int idx = (by + dy) * step + (bx + dx) * 3;
            img[idx]     = b_avg;
            img[idx + 1] = g_avg;
            img[idx + 2] = r_avg;
        }
    }
}
"""

def aplicar_pixelado_3d(img_original):
    context = device.make_context()
    try:
        h, w = img_original.shape[:2]
        escala = 1242 / w
        img = cv2.resize(img_original, (1242, int(h * escala)))
        h, w = img.shape[:2]
        img_bytes = img.flatten().astype(np.uint8)

        d_img = cuda.mem_alloc(img_bytes.nbytes)
        cuda.memcpy_htod(d_img, img_bytes)

        mod = SourceModule(cuda_kernel)
        func = mod.get_function("pixelar")

        grid_x = (w + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_y = (h + BLOCK_SIZE - 1) // BLOCK_SIZE

        func(d_img, np.int32(w), np.int32(h), np.int32(img.strides[0]),
            np.int32(BLOCK_SIZE), block=(1,1,1), grid=(grid_x, grid_y))

        cuda.memcpy_dtoh(img_bytes, d_img)
        img_pixelado = img_bytes.reshape((h, w, 3))

        for y in range(0, h, BLOCK_SIZE):
            for x in range(0, w, BLOCK_SIZE):
                cv2.rectangle(img_pixelado, (x, y), (x + BLOCK_SIZE, y + BLOCK_SIZE), (0, 0, 0), 1, lineType=cv2.LINE_AA)
                cv2.line(img_pixelado, (x, y + BLOCK_SIZE), (x + BLOCK_SIZE, y + BLOCK_SIZE), (200, 200, 200), 1, lineType=cv2.LINE_AA)
                cv2.line(img_pixelado, (x + BLOCK_SIZE, y), (x + BLOCK_SIZE, y + BLOCK_SIZE), (200, 200, 200), 1, lineType=cv2.LINE_AA)

        return img_pixelado
    finally:
        context.pop()