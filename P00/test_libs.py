# Verifica la existencia de varias librerias de Python y muestra sus versiones.
# (c) MJMJ

import sys

packages = {
    "numpy": "numpy",
    "opencv": "cv2",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "Pillow": "PIL",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
}

print("Python:", sys.version.split()[0])
print("-" * 50)

for name, module_name in packages.items():
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "versi�n no disponible")
        print(f"[OK] {name:15s} {version}")
    except ImportError:
        print(f"[--] {name:15s} NO INSTALADO")
        