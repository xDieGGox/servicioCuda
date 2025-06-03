# UPSGlam 2.0 - Servicio de Procesamiento de Imágenes con PyCUDA 

## Instalacion básica
Clonamos el proyecto:
```
git clone https://github.com/xDieGGox/servicioCuda
```
Instalamos los requerimientos, es opcional crear un entorno virtual, e instalar PyCUDA, flask, flask-cors, opencv-python, numpy.

Luego ingresamos en la ruta del proyecto clonado y lo ejecutamos con:
```
python app.py
```

## Descripción General

Este componente forma parte del proyecto **UPSGlam 2.0**, una plataforma social para compartir imágenes con diferentes filtros procesados en paralelo, este servicio integra tecnologías de computación paralela con **CUDA**, arquitectura de microservicios reactivos (**Spring WebFlux**), aplicación móvil en **Flutter** y almacenamiento en la nube (**Firebase**).

El presente servicio está desarrollado en **Python** usando **Flask** para crear un servidor web que expone múltiples filtros de procesamiento de imágenes aplicados con **PyCUDA**, aprovechando el paralelismo masivo de la tarjeta gráfica o GPU. Este servicio es consumido directamente desde una aplicación reactiva construido en Spring con WebFlux, enviando y recibiendo imágenes codificadas en **Base64** a través de los endpoints.

---

## ⚙️ Tecnologías Utilizadas

- **Python 3.9+**
- **Flask** - Framework para servicios web y exposición de enpoints.
- **PyCUDA** - Procesamiento paralelo en GPU para procesamiento de imagenes.
- **OpenCV** - Para lectura, decodificación y redimensionamiento de imágenes procesadas.
- **Flask-CORS** - Para permitir el consumo desde otras plataformas como WebFlux (Spring).
- **Docker** Este de aqui existe un archivo de docker el cual es para levantar el servicio, pero posteriormente verémos como se lo hace desde un docker compose para levantar los servicios en conjunto.
- **Spring WebFlux** - Cliente reactivo que consume esta API desde el microservicio de publicaciones.

---

## 🎨 Funcionalidad Principal

El servicio recibe imágenes en **formato base64** (desde WebFlux), las decodifica, redimensiona a un ancho estándar de `1242px` con OpenCV y aplica filtros en GPU usando **kernels CUDA**. Finalmente, responde con la imagen procesada también en base64.

---

## Endpoints expuestos

| Endpoint | Descripción | Tipo de Filtro |
|---------|-------------|----------------|
| `/gaussiano` | Aplica un filtro gaussiano usando convolución paralela. | Suavizado |
| `/laplaciano` | Filtro laplaciano personalizado para detección de bordes. | Detección de bordes |
| `/gabor` | Filtro de texturas mediante convolución de núcleos de Gabor. | Textura / Segmentación |
| `/pixel` | Efecto de pixelado artístico en bloques. | Estilización |
| `/rainbow` | Detección de bordes con superposición de fondo "arcoiris.jpg". | Efecto artístico |
| `/ups` | Filtro especial con diseño institucional de la Universidad Politécnica Salesiana. | Creativo |
| `/mixto` | Aplica dos filtros distintos a mitades de una misma imagen, estos filtros son el de arcoirid y el laplaciano de momento | Combinado |

---

## Ejemplo de Request (JSON)

```json
POST /gaussiano
Content-Type: application/json

{
  "image": "<cadena_base64_de_una_imagen_enviada_desde_spring_webflux>"
}**
```

## Arquitectura del Servicio
- **Redimensionamos:** Antes de aplicar el filtro, se estandariza el ancho de la imagen para mejorar el rendimiento del kernel CUDA, y el alto se deja para ajustarse automáticamente.

- **Filtro de convolución con CUDA:** Cada filtro tiene se procesa mediante el kernel de CUDA, aqui dependiendo del filtro seleccionado se aplica uno u otro para poder obtener la imagen y enviarla como respuesta e base64.

- **Ejecución de Bloques**: Se ejecutan con tamaños de bloque típicos como 32x32 para optimizar la utilización de hilos GPU, es decir que cada imagen se procesa con 1024 hilos para mejorar la eficiencia y poder obtener estas en un lapso de tiempo mínimo.

- **Separación de módulos:** Cada filtro se encuentra encapsulado en su propio archivo (filtro_gauss.py, filtro_pixel.py, etc.) para modularidad y mantenimiento del código siguiendo buenas prácticas.

## Dockerización
En este archivo vamos a encontrar un archivo Dockerfile con el cual a ejecutarlo tendremos una imagen para poder levantar este servicio de forma individual mediante los siguientes comandos
Contruimos la imagen con:
```
docker build -t servicio-cuda .  
```
Levantamos el servicio con:
```
docker run --rm -it --gpus all -p 5000:5000 servicio-cuda    
```

## Repositorios relacionados
Este punto es el Frontend del sistema, el cual utilizara el usuario final.
Repositorios del resto de arquitectura:

*Webflux:* https://github.com/dfao2003/api-webflux

*Flutter:* https://github.com/Chris-Liter/upsglam2 