import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os

cuda.init()
device = cuda.Device(0)

# CUDA: Filtro de suavizado
cuda_kernel = """
__global__ void filtro_artistico(unsigned char* img, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < w-1 && y >= 1 && y < h-1) {
        int idx = y * w + x;
        int sum = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += img[(y+dy)*w + (x+dx)];
            }
        }
        out[idx] = sum / 9; // Suavizado
    }
}
"""

def detectar_rostros_y_ojos(imagen_color):
    gray = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    rostros = face_cascade.detectMultiScale(gray, 1.3, 5)
    ojos_todos = []

    for (x, y, w, h) in rostros:
        roi_gray = gray[y:y+h, x:x+w]
        ojos = eye_cascade.detectMultiScale(roi_gray)
        ojos_abs = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in ojos]
        ojos_todos.extend(ojos_abs)

    return rostros, ojos_todos

def aplicar_convolucion_en_region(img_gray, x, y, w, h, context):
    roi = img_gray[y:y+h, x:x+w].copy()
    h_roi, w_roi = roi.shape
    roi_flat = roi.flatten().astype(np.uint8)
    salida = np.zeros_like(roi_flat)

    d_img = cuda.mem_alloc(roi_flat.nbytes)
    d_out = cuda.mem_alloc(salida.nbytes)

    cuda.memcpy_htod(d_img, roi_flat)
    cuda.memcpy_htod(d_out, salida)

    mod = SourceModule(cuda_kernel)
    func = mod.get_function("filtro_artistico")

    block = (32, 32, 1)
    grid = ((w_roi + 31)//32, (h_roi + 31)//32)
    func(d_img, d_out, np.int32(w_roi), np.int32(h_roi), block=block, grid=grid)

    cuda.memcpy_dtoh(salida, d_out)
    resultado = salida.reshape(h_roi, w_roi)

    d_img.free()
    d_out.free()
    return resultado

def colocar_overlay(base, overlay, x, y):
    h, w = overlay.shape[:2]
    for i in range(h):
        for j in range(w):
            if y + i >= base.shape[0] or x + j >= base.shape[1]:
                continue
            alpha = overlay[i, j, 3] / 255.0
            base[y + i, x + j] = (
                alpha * overlay[i, j, :3] + (1 - alpha) * base[y + i, x + j]
            ).astype(np.uint8)

def filtro_ups(imagen_color):
    context = device.make_context()
    try:
        # Cargar overlay desde la raíz
        overlay_path = os.path.join(os.path.dirname(__file__), "ups.png")
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None or overlay_img.shape[2] != 4:
            raise ValueError("La imagen 'ups.png' debe existir y tener canal alfa (RGBA)")

        rostros, ojos = detectar_rostros_y_ojos(imagen_color)
        img_gray = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in rostros:
            region_convol = aplicar_convolucion_en_region(img_gray, x, y, w, h, context)
            imagen_color[y:y+h, x:x+w, 0] = region_convol
            imagen_color[y:y+h, x:x+w, 1] = region_convol
            imagen_color[y:y+h, x:x+w, 2] = region_convol

        for (x, y, w, h) in ojos:
            mini_overlay = cv2.resize(overlay_img, (w, h))
            colocar_overlay(imagen_color, mini_overlay, x, y)

        return imagen_color
    finally:
        context.pop()
