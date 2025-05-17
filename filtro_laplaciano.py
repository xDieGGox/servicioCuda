import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
device = cuda.Device(0)

def aplicar_filtro_laplaciano(imagen_color, tamaño=9, block_x=32, block_y=32):
    context = device.make_context()
    try:
        # Convertir imagen a escala de grises y float32
        imagen_gray = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
        imagen_float = imagen_gray.astype(np.float32)

        altura, ancho = imagen_gray.shape

        # Crear máscara laplaciana
        laplaciana = -1 * np.ones((tamaño, tamaño), dtype=np.float32)
        centro = tamaño // 2
        laplaciana[centro, centro] = (tamaño * tamaño) - 1
        laplaciana_flat = laplaciana.flatten()

        salida_gpu = np.zeros_like(imagen_float)

        # Compilar kernel CUDA aquí (después de crear contexto)
        mod = SourceModule("""
        __global__ void filtro_laplaciano(float *imagen, float *mascara, float *salida, int ancho, int alto, int offset, int tamano_mascara) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= offset && x < (ancho - offset) && y >= offset && y < (alto - offset)) {
                float valor = 0.0;
                for (int i = -offset; i <= offset; i++) {
                    for (int j = -offset; j <= offset; j++) {
                        int idx = (i + offset) * tamano_mascara + (j + offset);
                        valor += imagen[(y + i) * ancho + (x + j)] * mascara[idx];
                    }
                }
                salida[y * ancho + x] = fminf(fmaxf(valor, 0.0), 255.0);
            }
        }
        """)

        # Reservar memoria
        imagen_gpu = cuda.mem_alloc(imagen_float.nbytes)
        salida_gpu_mem = cuda.mem_alloc(salida_gpu.nbytes)
        mascara_gpu = cuda.mem_alloc(laplaciana_flat.nbytes)

        cuda.memcpy_htod(imagen_gpu, imagen_float)
        cuda.memcpy_htod(mascara_gpu, laplaciana_flat)

        # Configurar bloque y grilla
        offset = tamaño // 2
        block_size = (block_x, block_y, 1)
        grid_size = ((ancho + block_x - 1) // block_x, (altura + block_y - 1) // block_y)

        # Ejecutar kernel
        filtro_laplaciano_gpu = mod.get_function("filtro_laplaciano")
        filtro_laplaciano_gpu(imagen_gpu, mascara_gpu, salida_gpu_mem,
                              np.int32(ancho), np.int32(altura),
                              np.int32(offset), np.int32(tamaño),
                              block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(salida_gpu, salida_gpu_mem)
        resultado_uint8 = salida_gpu.astype(np.uint8)
        return cv2.cvtColor(resultado_uint8, cv2.COLOR_GRAY2BGR)

    finally:
        context.pop()
