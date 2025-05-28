from flask import Flask, request, jsonify
from filtro_cuda import generar_mascara_gaussiana, aplicar_filtro_cuda
from filtro_pixel import aplicar_pixelado_3d
from filtro_arcoiris import filtro_bordes_con_fondo
from filtro_laplaciano import aplicar_filtro_laplaciano
from filtro_ups import filtro_ups
from filtro_gabor import aplicar_filtro_gabor
import numpy as np
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#Parámetros fijos
TAMANIO_MASCARA = 15
SIGMA = 8.0
BLOCK_X = 32
BLOCK_Y = 32
ANCHO_OBJETIVO = 1242  #medida a redimensionar la imagen de entrada

def redimensionar_imagen(img, ancho_objetivo=1242):
    h, w = img.shape[:2]
    escala = ancho_objetivo / w
    nuevo_tam = (ancho_objetivo, int(h * escala))
    return cv2.resize(img, nuevo_tam), nuevo_tam

@app.route("/gaussiano", methods=["POST"])
def aplicar_gaussiano():
    #if 'image' not in request.files:
        #return jsonify({"error": "No se envió una imagen"}), 400

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No se envió una imagen"}), 400

        imagen_base64 = data['image']
        imagen_bytes = base64.b64decode(imagen_base64)
        #archivo = request.files['image']
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)
        imagen_original = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen_original is None:
            return jsonify({"error": "Imagen no válida"}), 400

        #Redimensionar la imagen antes de aplicar el filtro
        imagen, nuevo_tam = redimensionar_imagen(imagen_original, ANCHO_OBJETIVO)

        mascara = generar_mascara_gaussiana(TAMANIO_MASCARA, SIGMA)
        salida = aplicar_filtro_cuda(imagen, mascara, BLOCK_X, BLOCK_Y)

        _, buffer = cv2.imencode('.jpg', salida)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "imagen": base64_img,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/pixel", methods=["POST"])
def aplicar_pixel():
    #if 'image' not in request.files:
        #return jsonify({"error": "No se envió una imagen"}), 400

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({"error": "No se envió una imagen"}), 400

        # Decodificar base64 a bytes
        imagen_base64 = data['image']
        imagen_bytes = base64.b64decode(imagen_base64)
        #archivo = request.files['image']
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)
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

    #if 'image' not in request.files:
        #return jsonify({"error": "Falta el archivo 'image'"}), 400

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({"error": "No se envió una imagen"}), 400

        # Decodificar base64 a bytes
        imagen_base64 = data['image']
        imagen_bytes = base64.b64decode(imagen_base64)

        # Leer imagen principal desde el request
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)
        img = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

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

@app.route("/ups", methods=["POST"])
def aplicar_ups():
    #if 'image' not in request.files:
        #return jsonify({"error": "No se envió una imagen"}), 400

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({"error": "No se envió una imagen"}), 400

        # Decodificar base64 a bytes
        imagen_base64 = data['image']
        imagen_bytes = base64.b64decode(imagen_base64)
        #archivo = request.files['image']
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        imagen_redim, _ = redimensionar_imagen(imagen)

        resultado = filtro_ups(imagen_redim)

        _, buffer = cv2.imencode('.jpg', resultado)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"imagen": base64_img})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/laplaciano", methods=["POST"])
def aplicar_laplaciano():
    #if 'image' not in request.files:
        #return jsonify({"error": "No se envió una imagen"}), 400

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({"error": "No se envió una imagen"}), 400

        # Decodificar base64 a bytes
        imagen_base64 = data['image']
        imagen_bytes = base64.b64decode(imagen_base64)
        #archivo = request.files['image']
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        #Redimensionar
        imagen_redim, _ = redimensionar_imagen(imagen)

        #Aplicar filtro
        resultado = aplicar_filtro_laplaciano(imagen_redim)

        _, buffer = cv2.imencode('.jpg', resultado)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"imagen": base64_img})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/gabor", methods=["POST"])
def aplicar_gabor():
    print("Gabor")

    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({"error": "No se envió una imagen"}), 400

        # Decodificar base64 a bytes
        imagen_base64 = data['image']
        imagen_bytes = base64.b64decode(imagen_base64)
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)

        # Decodificar bytes a imagen con OpenCV
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        # Redimensionar
        imagen_redim, _ = redimensionar_imagen(imagen)

        # Aplicar filtro
        resultado = aplicar_filtro_gabor(imagen_redim)
        print("Proceso")

        # Codificar la imagen procesada a base64
        _, buffer = cv2.imencode('.jpg', resultado)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"imagen": base64_img})

    except Exception as e:
        print("Error en aplicar_gabor:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/mixto", methods=["POST"])
def aplicar_mixto():
    #if 'image' not in request.files:
        #return jsonify({"error": "No se envió una imagen"}), 400
    #if 'filtro1' not in request.form or 'filtro2' not in request.form:
        #return jsonify({"error": "Debes enviar los filtros 'filtro1' y 'filtro2'"}), 400

    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "Debes enviar 'image'"}), 400

        filtro1 = "rainbow"
        filtro2 = "pixel"

        
        imagen_base64 = data["image"]
        imagen_bytes = base64.b64decode(imagen_base64)
        #archivo = request.files['image']
        archivo_np = np.frombuffer(imagen_bytes, np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)
        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        #Redimensionar a 1242px de ancho
        imagen_redim, _ = redimensionar_imagen(imagen, ANCHO_OBJETIVO)
        h, w, _ = imagen_redim.shape

        mitad_px = ANCHO_OBJETIVO // 2  # 1242 / 2 = 621
        izquierda = np.ascontiguousarray(imagen_redim[:, :mitad_px])
        derecha = np.ascontiguousarray(imagen_redim[:, mitad_px:])

        # Asegurar que son 3 canales
        if len(izquierda.shape) == 2:
            izquierda = cv2.cvtColor(izquierda, cv2.COLOR_GRAY2BGR)
        if len(derecha.shape) == 2:
            derecha = cv2.cvtColor(derecha, cv2.COLOR_GRAY2BGR)

        #Aplicar filtros según parámetros
        def aplicar(filtro, img):
            if filtro == 'gaussiano':
                mascara = generar_mascara_gaussiana(3, 0.8)
                return aplicar_filtro_cuda(img, mascara, BLOCK_X, BLOCK_Y)[0]
            elif filtro == 'pixel':
                return aplicar_pixelado_3d(img)
            elif filtro == 'laplaciano':
                return aplicar_filtro_laplaciano(img)
            elif filtro == 'rainbow':
                import os
                fondo_path = os.path.join(os.path.dirname(__file__), "arcoiris.jpg")
                fondo = cv2.imread(fondo_path)
                if fondo is None:
                    raise Exception("No se encontró el fondo arcoiris.jpg")
                return filtro_bordes_con_fondo(img, fondo)

        mitad_izq = aplicar(filtro1, izquierda)
        mitad_der = aplicar(filtro2, derecha)

        #Asegurar que ambas mitades tengan misma altura y 3 canales
        if len(mitad_izq.shape) == 2:
            mitad_izq = cv2.cvtColor(mitad_izq, cv2.COLOR_GRAY2BGR)
        if len(mitad_der.shape) == 2:
            mitad_der = cv2.cvtColor(mitad_der, cv2.COLOR_GRAY2BGR)

        # Igualar altura si algún filtro lo modificó
        #target_height = min(mitad_izq.shape[0], mitad_der.shape[0])
        #mitad_izq = cv2.resize(mitad_izq, (mitad_px, target_height))
        #mitad_der = cv2.resize(mitad_der, (ANCHO_OBJETIVO - mitad_px, target_height))
        
        # Forzar altura original y ancho exacto de cada mitad
        mitad_izq = cv2.resize(mitad_izq, (mitad_px, h))
        mitad_der = cv2.resize(mitad_der, (ANCHO_OBJETIVO - mitad_px, h))

        #Crear lienzo final y unir mitades
        salida_final = np.hstack((mitad_izq, mitad_der))

        _, buffer = cv2.imencode('.jpg', salida_final)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        print("MITAD IZQ:", mitad_izq.shape)
        print("MITAD DER:", mitad_der.shape)

        return jsonify({
            "imagen": base64_img
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
