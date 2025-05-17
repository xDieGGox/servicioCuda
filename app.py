from flask import Flask, request, jsonify
from filtro_cuda import generar_mascara_gaussiana, aplicar_filtro_cuda
from filtro_pixel import aplicar_pixelado_3d
from filtro_arcoiris import filtro_bordes_con_fondo
import numpy as np
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🎯 Parámetros fijos
TAMANIO_MASCARA = 15
SIGMA = 8.0
BLOCK_X = 32
BLOCK_Y = 32
ANCHO_OBJETIVO = 1242  # Nuevo parámetro para redimensionar

def redimensionar_imagen(img, ancho_objetivo=1242):
    h, w = img.shape[:2]
    escala = ancho_objetivo / w
    nuevo_tam = (ancho_objetivo, int(h * escala))
    return cv2.resize(img, nuevo_tam), nuevo_tam

@app.route("/gaussiano", methods=["POST"])
def aplicar_gaussiano():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    try:
        archivo = request.files['image']
        archivo_np = np.frombuffer(archivo.read(), np.uint8)
        imagen_original = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen_original is None:
            return jsonify({"error": "Imagen no válida"}), 400

        #Redimensionar la imagen antes de aplicar el filtro
        imagen, nuevo_tam = redimensionar_imagen(imagen_original, ANCHO_OBJETIVO)

        mascara = generar_mascara_gaussiana(TAMANIO_MASCARA, SIGMA)
        salida, tiempo, grid_x, grid_y = aplicar_filtro_cuda(imagen, mascara, BLOCK_X, BLOCK_Y)

        _, buffer = cv2.imencode('.jpg', salida)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "imagen": base64_img,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/pixel", methods=["POST"])
def aplicar_pixel():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    try:
        archivo = request.files['image']
        archivo_np = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        imagen_pixelada = aplicar_pixelado_3d(imagen)

        _, buffer = cv2.imencode('.jpg', imagen_pixelada)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "imagen": base64_img,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rainbow", methods=["POST"])
def aplicar_rainbow():
    import os

    if 'image' not in request.files:
        return jsonify({"error": "Falta el archivo 'image'"}), 400

    try:
        # Leer imagen principal desde el request
        img_bytes = np.frombuffer(request.files['image'].read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Leer imagen de fondo desde archivo local
        fondo_path = os.path.join(os.path.dirname(__file__), "arcoiris.jpg")
        if not os.path.exists(fondo_path):
            return jsonify({"error": "La imagen de fondo 'fondo_arcoiris.jpg' no fue encontrada en el directorio del proyecto"}), 400

        bg = cv2.imread(fondo_path)

        if img is None or bg is None:
            return jsonify({"error": "Error al cargar la imagen principal o el fondo"}), 400

        # Aplicar el filtro
        resultado = filtro_bordes_con_fondo(img, bg)
        _, buffer = cv2.imencode('.jpg', resultado)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"imagen": base64_img})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
