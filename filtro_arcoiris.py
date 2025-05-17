import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Inicialización global
cuda.init()
device = cuda.Device(0)

cuda_kernel = """
__global__ void detectar_bordes(unsigned char* gray, unsigned char* salida, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * w + x;

    if (x > 1 && y > 1 && x < w - 2 && y < h - 2) {
        int gx = 
            -1 * gray[(y-1)*w + (x-1)] + 1 * gray[(y-1)*w + (x+1)] +
            -2 * gray[(y)*w + (x-1)] + 2 * gray[(y)*w + (x+1)] +
            -1 * gray[(y+1)*w + (x-1)] + 1 * gray[(y+1)*w + (x+1)];

        int gy = 
            -1 * gray[(y-1)*w + (x-1)] + -2 * gray[(y-1)*w + (x)] + -1 * gray[(y-1)*w + (x+1)] +
             1 * gray[(y+1)*w + (x-1)] +  2 * gray[(y+1)*w + (x)] +  1 * gray[(y+1)*w + (x+1)];

        int mag = sqrtf(float(gx * gx + gy * gy));
        if (mag > 100) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int idx2 = (y + dy) * w + (x + dx);
                    salida[idx2] = 255;
                }
            }
        }
    }
}
"""

def redimensionar_imagen(img, ancho_objetivo=1242):
    h, w = img.shape[:2]
    escala = ancho_objetivo / w
    nuevo_tam = (ancho_objetivo, int(h * escala))
    return cv2.resize(img, nuevo_tam), nuevo_tam

def procesar_bordes_cuda(imagen_gray, context):
    h, w = imagen_gray.shape
    size = w * h

    salida_cpu = np.zeros_like(imagen_gray)

    gray_gpu = cuda.mem_alloc(size)
    output_gpu = cuda.mem_alloc(size)

    cuda.memcpy_htod(gray_gpu, imagen_gray)
    cuda.memcpy_htod(output_gpu, salida_cpu)

    mod = SourceModule(cuda_kernel)
    func = mod.get_function("detectar_bordes")

    block = (32, 32, 1)
    grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])

    func(gray_gpu, output_gpu, np.int32(w), np.int32(h), block=block, grid=grid)

    cuda.memcpy_dtoh(salida_cpu, output_gpu)

    gray_gpu.free()
    output_gpu.free()

    return salida_cpu

def fusionar_bordes_con_fondo(bordes, fondo, tamaño):
    fondo_redim = cv2.resize(fondo, tamaño)
    bordes_color = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
    return np.where(bordes_color == 255, fondo_redim, 0)

def filtro_bordes_con_fondo(imagen_bgr, fondo_bgr):
    context = device.make_context()
    try:
        imagen_redim, tamaño = redimensionar_imagen(imagen_bgr)
        fondo_redim = cv2.resize(fondo_bgr, tamaño)

        imagen_gray = cv2.cvtColor(imagen_redim, cv2.COLOR_BGR2GRAY)
        bordes = procesar_bordes_cuda(imagen_gray, context)

        resultado = fusionar_bordes_con_fondo(bordes, fondo_redim, tamaño)
        return resultado
    finally:
        context.pop()
