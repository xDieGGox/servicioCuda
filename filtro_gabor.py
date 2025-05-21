import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
device = cuda.Device(0)

def generate_gabor_kernel(ksize, sigma, theta, lambd, psi, gamma):
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    half = ksize // 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            x_theta = x * cos_theta + y * sin_theta
            y_theta = -x * sin_theta + y * cos_theta
            gauss = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
            wave = np.cos(2 * np.pi * x_theta / lambd + psi)
            kernel[y + half, x + half] = gauss * wave
    return kernel

# Kernel CUDA para Gabor en color
kernel_gabor_code = """
__global__ void applyGaborCUDA(uchar3* input, uchar3* output, float* kernel, 
                               int ksize, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = ksize / 2;

    if (x >= half && y >= half && x < width - half && y < height - half) {
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int imgX = x + kx;
                int imgY = y + ky;
                int idx = imgY * width + imgX;
                int kidx = (ky + half) * ksize + (kx + half);
                uchar3 pixel = input[idx];
                float weight = kernel[kidx];
                sumR += weight * pixel.x;
                sumG += weight * pixel.y;
                sumB += weight * pixel.z;
            }
        }
        int outIdx = y * width + x;
        output[outIdx].x = min(max(int(sumR), 0), 255);
        output[outIdx].y = min(max(int(sumG), 0), 255);
        output[outIdx].z = min(max(int(sumB), 0), 255);
    }
}
"""

def aplicar_filtro_gabor(imagen_color, tamaño_kernel=9, block_x=32, block_y=32):
    context = device.make_context()
    try:
        # Convertir imagen a RGB (esperado por el kernel)
        imagen_rgb = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB)
        height, width = imagen_rgb.shape[:2]

        imagen_flat = imagen_rgb.reshape(-1, 3).astype(np.uint8)
        resultado_flat = np.empty_like(imagen_flat)

        # Crear kernel Gabor personalizado
        gabor_kernel = generate_gabor_kernel(tamaño_kernel, 5.0, np.pi/4, 10.0, 2.0, 2.0)
        gabor_kernel_flat = gabor_kernel.flatten()

        # Compilar kernel CUDA
        mod = SourceModule(kernel_gabor_code)
        apply_gabor = mod.get_function("applyGaborCUDA")

        # Reservar memoria GPU
        d_input = cuda.mem_alloc(imagen_flat.nbytes)
        d_output = cuda.mem_alloc(resultado_flat.nbytes)
        d_kernel = cuda.mem_alloc(gabor_kernel_flat.nbytes)

        # Copiar datos a GPU
        cuda.memcpy_htod(d_input, imagen_flat)
        cuda.memcpy_htod(d_kernel, gabor_kernel_flat)

        # Configurar ejecución CUDA
        block_size = (block_x, block_y, 1)
        grid_size = ((width + block_x - 1) // block_x, (height + block_y - 1) // block_y)

        # Ejecutar kernel
        apply_gabor(d_input, d_output, d_kernel,
                    np.int32(tamaño_kernel), np.int32(width), np.int32(height),
                    block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(resultado_flat, d_output)
        resultado_rgb = resultado_flat.reshape((height, width, 3))

        return cv2.cvtColor(resultado_rgb, cv2.COLOR_RGB2BGR)

    finally:
        context.pop()
