# VisIAqp — multicámara y robustez de transmisiones

## Implementación

El modo cámaras utiliza un supervisor local con procesos independientes de Windows (`spawn`). Cada cámara tiene su propia instancia de YOLO y su propio seguimiento. El límite inicial es de dos cámaras simultáneas; las restantes esperan una plaza. Una transmisión continua ocupa su plaza hasta que se detiene o cambia su configuración. No se incorporaron Redis, Celery ni servicios externos.

Cada diez segundos se vuelve a consultar el catálogo. Las cámaras eliminadas o deshabilitadas se detienen y los cambios de fuente, polígono o intervalo reinician únicamente la cámara afectada. Un error al consultar el catálogo no equivale a una lista vacía. Las ventanas de depuración de OpenCV quedan deshabilitadas en los procesos del supervisor.

La API reserva la cámara antes de cargar el modelo o abrir la fuente. Las reservas de transmisiones se serializan con un bloqueo de fila de MySQL; dos solicitudes simultáneas para una misma cámara dejan una ejecución activa y una respuesta 409. Una cámara deshabilitada no admite una reserva nueva. Esto protege el identificador de cámara: registrar una misma URL bajo varios identificadores sigue requiriendo control administrativo.

La captura en vivo lee continuamente y conserva únicamente el frame más reciente. El tiempo del evento se aproxima con la hora de recepción local, no con la hora posterior a la inferencia. Los tiempos de apertura y lectura de fuentes de red se limitan a diez segundos mediante FFmpeg. Una caída termina la ejecución como error y el trabajador vuelve a intentar abrirla. Cada reconexión crea una ejecución y un seguimiento nuevos. Si la captura no responde al cierre, su proceso termina para que el supervisor lo recree, evitando acumular hilos de captura bloqueados.

## Actividad y recuperación

Se añade `PATCH /api/procesamientos/{id}/actividad`, protegido por la misma API key del motor. El motor envía actividad al comenzar y aproximadamente cada diez segundos mientras procesa imágenes. La API también renueva actividad al aceptar resultados.

Las ejecuciones vencen tras 120 segundos sin actividad, configurables mediante `MOTOR_INACTIVIDAD_SEG` en el backend, entre 30 y 3600. El vencimiento se evalúa al consultar ejecuciones, buscar videos pendientes, reservar una cámara o comprobar la vigencia de una ejecución. No es una tarea programada independiente. La migración concede un plazo inicial a las ejecuciones que ya estaban procesando.

Un trabajador vencido no puede publicar resultados ni renovar su reserva. Tampoco puede convertir su estado de error en completado mediante un cierre tardío. El motor deja de procesar si la reserva no se confirma o si falla la señal de actividad; no continúa haciendo inferencia sin una reserva reconocida por el backend.

Un video abandonado queda en error y conserva sus resultados parciales. No se reencola desde el comienzo automáticamente, porque eso duplicaría observaciones. La exclusión de reprocesamientos del historial sigue pendiente; no se recomienda repetir masivamente videos con resultados parciales.

## Mediciones y ventanas

El backend expone en las ejecuciones `ultima_actividad`, `ultima_muestra`, `fps_procesamiento`, `retraso_seg` e `inactividad_limite_seg`. Monitoreo muestra el estado del motor y esas mediciones. Una cámara habilitada no se presenta automáticamente como conectada.

`retraso_seg` es la edad desde la recepción local hasta terminar de procesar la imagen. No mide el retraso entre el sensor y el equipo: una cámara, la red o el decodificador todavía pueden entregar contenido antiguo. No se implementa sincronización de relojes de sensores ni lectura de sus marcas temporales.

`intervalo_muestreo_seg` llega ahora al motor y define la ventana de reporte de cada cámara, tanto para el catálogo de cámaras como para videos solicitados. La interfaz lo llama **Ventana de reporte**. `target_fps` controla por separado la frecuencia de imágenes analizadas. Si entre dos muestras en vivo pasan más de cinco segundos, se cierra la ventana previa y se reinicia el seguimiento; no se extiende la ocupación a través de ese hueco.

Cada ventana nueva lleva una `clave_ventana` opcional. La API devuelve el mismo resultado ante reenvíos simultáneos de una misma ventana activa; si se reutiliza la clave con datos distintos responde 409. Los motores antiguos que no envían clave siguen siendo compatibles, pero no tienen esta protección contra reenvíos.

Los envíos HTTP siguen usando reintentos acotados. Durante esos reintentos pueden dejar de analizarse imágenes, aunque la captura mantiene el frame más reciente. Los resultados no enviados se registran en archivos locales separados por proceso, con su identificador de procesamiento y clave de ventana. No se implementó una cola de reenvío automática ni una recuperación completa de esos archivos hacia ejecuciones cerradas.

## Configuración aplicada

En `C:\Users\Wilson\Desktop\VisIAqp-ai\visiaqp-ai\config\config.base.yaml`:

```yaml
processing:
  resize_width: 960
  cpu_threads: 2
  read_timeout_seconds: 10
  max_gap_seconds: 5

multicamara:
  max_camaras: 2
  actualizar_seg: 10
  reintento_seg: 5
  actividad_seg: 10
  cierre_seg: 20
```

En desarrollo se mantiene CPU y se ajusta `target_fps` de 5 a 2. La configuración de producción existente continúa usando CUDA y 5 FPS; no fue validada en este equipo. La frecuencia de actividad del motor debe ser menor que el plazo de vencimiento configurado en el backend.

## Rendimiento comprobado

Se ejecutó YOLO real con `TownCentreXVID.mp4`, ancho de 960 píxeles, 60 imágenes analizadas por cámara y dos hilos de PyTorch por proceso. El ensayo no envió resultados a la API. El equipo informa 20 procesadores lógicos.

| Procesos simultáneos | FPS efectivos por proceso | Memoria residente por proceso |
|---|---|---|
| 1 | 4,156 | 512,4 MB |
| 2 | 3,344 y 3,342 | 502,4 y 501,8 MB |

Es una medición breve de procesamiento de video local, no una certificación de capacidad máxima ni una prueba prolongada con transmisiones reales. El primer ensayo coincidió parcialmente con otras verificaciones. El valor de desarrollo de 2 FPS deja margen respecto al ensayo de dos procesos, pero debe volver a medirse bajo la carga real del sistema.

No se ejecutó el ensayo de cuatro procesos: la comprobación previa encontró aproximadamente 0,5 GB de memoria disponible y lo detuvo antes de cargar modelos adicionales. Esto describe la memoria disponible en ese momento, no un máximo físico del equipo. No se ha medido la precisión de conteo al reducir la frecuencia de muestreo.

El script reproducible está en `C:\Users\Wilson\Desktop\VisIAqp-ai\visiaqp-ai\scripts\medir_capacidad.py`. Los resultados originales se adjuntan en `medicion-capacidad.json` junto a este documento.

## Pruebas y migración

- Backend: 73 pruebas aprobadas, incluidas solicitudes concurrentes reales sobre MySQL para reserva e idempotencia.
- Motor: 62 pruebas aprobadas y 2 omitidas por enlaces simbólicos no disponibles. Incluyen captura reciente, caída frente a fin de archivo, reconexión, aislamiento, límite de procesos, cambio de catálogo, fallo de modelo y captura que no responde al cierre.
- Frontend: 45 pruebas aprobadas, incluida la distinción entre cámara habilitada y actividad de transmisión.

Migración aplicada: `C:\Users\Wilson\Desktop\VisIAqp-backend\visiaqp-api\alembic\versions\d48b2f0137bc_actividad_motor.py`. Se verificó primero sobre una copia vacía del esquema real en `visiaqp_test`. Después se aplicó en `visiaqp`, puerto 3306, conservando las cantidades de registros de resultados, indicadores, procesamientos, cámaras y zonas.

Desde `C:\Users\Wilson\Desktop\VisIAqp-backend\visiaqp-api`:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Desde `C:\Users\Wilson\Desktop\VisIAqp-ai\visiaqp-ai`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts/medir_capacidad.py --video data/videos/TownCentreXVID.mp4 --frames 60 --camaras 1 2
```

Desde `C:\Users\Wilson\Desktop\VisIAqp`:

```powershell
.\node_modules\.bin\ng.cmd test --watch=false --browsers=ChromeHeadless --progress=false
```

Para cargar los cambios, reiniciar primero el backend y después el motor y el frontend. Para cámaras, desde la carpeta del motor:

```powershell
$env:VISIAQP_ENV = 'dev'
.\.venv\Scripts\python.exe main.py --modo camaras
```

El modo videos sigue ejecutándose por separado con `--modo videos`. No se iniciaron procesos permanentes ni se reiniciaron los que pudiera tener abiertos el usuario.

## Límites pendientes

No se probó una caída física de una cámara RTSP, ni funcionamiento continuo prolongado, ni GPU. Se comprobaron las rutas de fallo con capturas y notificaciones controladas en pruebas, y el aislamiento de procesos con inferencia real durante la medición. Los archivos sin FPS legibles ahora fallan explícitamente en lugar de inventar tiempos de evento a partir de la velocidad de procesamiento.

La aplicación continúa usando fechas sin zona horaria explícita. Se comprobó que Python y MySQL de este equipo coinciden en hora local UTC-5; el despliegue en otra zona requiere unificar ese contrato. La consolidación espacial de varias cámaras, los endpoints agregados de métricas, el dashboard ampliado y clima/calendario siguen pendientes. La vista actual consulta las últimas 500 ejecuciones; el resumen operativo completo corresponde a la siguiente fase de endpoints.
