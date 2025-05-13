from flask import Flask, request, jsonify
from filtro_cuda import generar_mascara_gaussiana, aplicar_filtro_cuda
import numpy as np
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/gaussiano", methods=["POST"])
def aplicar_gaussiano():
    if 'image' not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    try:
        tamaño = int(request.form.get("mask", 21))
        sigma = float(request.form.get("sigma", 10.0))
        block_x = int(request.form.get("block_x", 32))
        block_y = int(request.form.get("block_y", 32))

        archivo = request.files['image']
        archivo_np = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "Imagen no válida"}), 400

        mascara = generar_mascara_gaussiana(tamaño, sigma)
        salida, tiempo, grid_x, grid_y = aplicar_filtro_cuda(imagen, mascara, block_x, block_y)

        _, buffer = cv2.imencode('.jpg', salida)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "imagen": base64_img,
            "tiempo_ms": round(tiempo, 2),
            "mask_size": tamaño,
            "sigma": sigma,
            "block": {"x": block_x, "y": block_y},
            "grid": {"x": grid_x, "y": grid_y}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
