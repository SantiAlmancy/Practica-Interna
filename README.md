# Traductor de Lenguaje de Señas en Tiempo Real

## Instalación de Bibliotecas
Para instalar las bibliotecas necesarias, ejecute los siguientes comandos en la línea de comandos después de haber instalado Python:

```bash
pip install pathlib
pip install collections
pip install Preprocess
pip install Detection
pip install numpy
pip install tensorflow
pip install pyttsx3
pip install scikit-learn
pip install opencv-python
pip install mediapipe
pip install Pillow
pip install imutils
```

## Instalación del Proyecto

Descargar el proyecto desde GitHub es un proceso sencillo y se puede realizar siguiendo estos pasos detallados. Asegúrese de considerar que el proyecto es considerablemente pesado, con un tamaño total de 496 MB. Además, tenga en cuenta que la carpeta denominada Additional Trained Data contiene datos de señas grabadas que, debido a limitaciones de hardware, no se utilizan en el producto final. Puede optar por omitir la descarga de esta carpeta si así lo desea.

**Pasos para descargar el proyecto desde GitHub:**

1. **Acceso al Repositorio:** Abra su navegador web y acceda al repositorio del proyecto en GitHub. Puede encontrar el enlace en la sección de anexos (\ref{sec:anexos}).

2. **Selección de la Rama:** Asegúrese de estar en la rama correcta del proyecto. El proyecto se encuentra en la rama Master.

3. **Clonar el Repositorio:** En la página principal del repositorio, haga clic en el botón Code (Código). Copie la URL proporcionada.

4. **Abra la Terminal (o Git Bash en Windows):** Navegue al directorio donde desea almacenar el proyecto utilizando el comando cd.

5. **Ejecute el Comando de Clonación:** En la terminal, escriba el siguiente comando y presione Enter: `git clone URLC`. Asegúrese de reemplazar URLC con la URL que copió anteriormente.

6. **Descargar Datos Adicionales (Opcional):** Si opta por no descargar la carpeta Additional Trained Data, puede excluirla manualmente del proceso de clonación o eliminarla después de la descarga principal.

7. **Esperar a que Finalice la Descarga:** El proceso de clonación llevará algún tiempo debido al tamaño del proyecto. Espere a que se complete.

8. **Acceder al Proyecto Descargado:** Una vez que la descarga haya finalizado, navegue al directorio del proyecto utilizando el comando cd.

## Requerimientos de Hardware

Los requerimientos de hardware para ejecutar el proyecto de traductor de lenguaje de señas son fundamentales para garantizar un rendimiento óptimo. A continuación, se detallan los requisitos mínimos recomendados:

- **Memoria RAM (RAM):** Se recomienda un mínimo de 8 GB de RAM para garantizar un rendimiento fluido durante la ejecución del proyecto, especialmente al entrenar modelos con bibliotecas como Keras.
- **Almacenamiento:** Para la descarga del proyecto, se necesita un espacio de almacenamiento mínimo de 266 MB (496 MB - 230 MB para la carpeta Additional Trained Data). Además, considere espacio adicional para el sistema operativo y otros archivos temporales.
- **Procesador (CPU):** Se recomienda un procesador de al menos 2 GHz para realizar operaciones computacionales de manera eficiente, especialmente durante el entrenamiento de modelos de redes neuronales.
- **Tarjeta Gráfica (GPU):** Aunque no es estrictamente necesario, el uso de una GPU compatible con CUDA puede acelerar significativamente el entrenamiento de modelos de aprendizaje profundo. Se recomienda una GPU NVIDIA con soporte para CUDA si está disponible.
- **Cámara:** Para la detección de señas mediante visión por computadora, se requiere una cámara integrada o conectada al sistema.
- **Salida de Sonido:** Para la funcionalidad de traducción de texto a voz, es necesario contar con una salida de sonido funcional, ya sea altavoces integrados, auriculares o cualquier otro dispositivo de salida de audio.

Estos son los requisitos esenciales para ejecutar el proyecto de manera efectiva. Asegúrese de contar con los controladores necesarios para los dispositivos de entrada y salida, como la cámara y la salida de sonido, para garantizar un funcionamiento adecuado de todas las funcionalidades del proyecto.
