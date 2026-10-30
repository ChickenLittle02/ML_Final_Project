# Parte 4 de 6 — Kendry: Datos y conjuntos de datos para el entrenamiento

**Tema general:** Monitoreo y gestión del espectro radioeléctrico mediante *Machine Learning*: detección de interferencias e identificación de espacios en blanco.  
**Responsable:** Kendry  
**Tipo de trabajo:** Estado del arte — revisión documental y comparación de conjuntos de datos.  
**Fecha de consulta y verificación documental:** 30 de octubre de 2026.  
**Estado de verificación:** se contrastaron los metadatos y descripciones con las páginas de los nueve repositorios originales y, para *Indoor Jamming* y Fraunhofer GNSS, con publicaciones o repositorios vinculados. No se verificaron los archivos de datos completos mediante descarga, lectura y cálculo de integridad.

## 4.1. Introducción y objetivo

La disponibilidad y la calidad de los datos condicionan el desarrollo de modelos de *Machine Learning* aplicados al espectro radioeléctrico. Para identificar una transmisión, detectar una interferencia o estimar si una banda está ocupada, no basta con disponer de un archivo de señales: es necesario conocer **qué se midió, cómo se midió, con qué resolución, bajo qué condiciones y qué información permite utilizarse como referencia o etiqueta**.

Esta sección examina conjuntos de datos públicos relevantes para las dos líneas del proyecto: **detección de interferencias** e **identificación de espacios en blanco** (*spectrum white spaces*). Su objetivo práctico es determinar qué materiales pueden aprovecharse **sin comprar ni operar equipos especializados de radio**. El análisis se basa en los registros públicos de los repositorios y en publicaciones asociadas; **no se han descargado, inspeccionado ni validado experimentalmente los archivos binarios completos de estos conjuntos**. En consecuencia, la accesibilidad que se indica se refiere a la existencia de fuentes y opciones de descarga publicadas, no a una garantía de que todos los enlaces, archivos y entornos de ejecución funcionen sin ajustes.

### Distinción fundamental entre los dos problemas

Una **interferencia radioeléctrica** se produce cuando una señal o emisión degrada, perturba o dificulta la recepción de otra señal de interés. En los datos, puede representarse mediante ejemplos de comunicación normal y de comunicación afectada por señales interferentes. Algunos conjuntos etiquetan directamente la clase de interferencia; otros muestran transmisiones simultáneas, pero **superposición no implica necesariamente interferencia perjudicial**.

Un **espacio en blanco espectral** es una región de tiempo y frecuencia que no presenta actividad detectable, o que resulta disponible para determinado esquema de acceso al espectro. Es imprescindible distinguir entre **ausencia de energía detectable**, **ausencia real de transmisores** y **autorización para transmitir**. La disponibilidad regulatoria depende del contexto y de las reglas de uso de la banda; una medición de potencia por sí sola no demuestra que transmitir esté permitido.

## 4.2. Tipos de datos empleados en la literatura

| Tipo de dato | Información que aporta | Utilidad principal | Limitación importante |
|---|---|---|---|
| **Potencia recibida por frecuencia / PSD** (*power spectral density*) | Niveles de potencia en distintos intervalos o *bins* de frecuencia, a menudo asociados a marcas temporales | Detección de actividad, construcción de mapas de ocupación y búsqueda de periodos aparentemente libres | Una potencia baja puede deberse a una señal débil o a limitaciones del receptor; no constituye una etiqueta perfecta de canal libre |
| **Registros binarios de ocupación** | Indicadores ocupado/libre por canal y por momento | Clasificación de ocupación y predicción temporal | Si se obtienen mediante umbrales, heredan los errores de ese umbral; pueden no distinguir usuarios primarios/secundarios |
| **Muestras complejas I/Q** (*in-phase/quadrature*) | Amplitud y fase de una señal muestreada; permiten reconstruir características temporales y espectrales | Detección de interferencia, identificación de señales y análisis fino de patrones RF | Pueden exigir mucho almacenamiento y procesamiento; una captura I/Q aislada no describe toda una banda ni su ocupación histórica |
| **Espectrogramas** | Representación bidimensional de energía o intensidad en tiempo y frecuencia | Localización de transmisiones, reconocimiento de patrones y detección de colisiones | Su resolución depende de la ventana y los parámetros del procesamiento; las imágenes no incluyen necesariamente etiquetas fiables |
| **Metadatos de medición** | Frecuencia central, ancho de banda, instante de captura, lugar, receptor, ganancia, tasa de muestreo, etc. | Interpretación, reproducibilidad y comparación de resultados | Metadatos incompletos dificultan la generalización y pueden impedir comparar campañas |

Un formato relevante es **SigMF (Signal Metadata Format)**, que permite almacenar muestras o mediciones junto con metadatos estructurados. No debe asumirse que todos los archivos SigMF contienen muestras I/Q: algunos repositorios lo emplean para **vectores de potencia espectral**. La especificación se encuentra en [SigMF](https://sigmf.org/) [1].

### Datos reales y simulados

**Importante sobre el término «datos reales»:** todos los conjuntos de la tabla corresponden a publicaciones o repositorios identificables, pero no todos son mediciones físicas. Hay **seis conjuntos basados en capturas/mediciones reales** (LOG-a-TEC; AERPAW Packapalooza; AERPAW nodos fijos; Fraunhofer GNSS; Indoor Jamming; NIST CBRS) y **tres conjuntos sintéticos** (RFSSC; Fraunhofer Extended Spectrogram; RadioML 2016.10A). La existencia del repositorio no equivale a verificar cada archivo ni su calidad científica.

Los **datos reales** son capturados en un entorno físico mediante sensores o radios definidos por software (SDR). Incorporan variabilidad propia de los equipos, propagación, interferentes y condiciones del entorno, pero suelen carecer de etiquetas exhaustivas de verdad terreno (*ground truth*). Los **datos simulados** se generan mediante modelos de señales y canales: permiten controlar la potencia, la ocupación y las etiquetas, pero el resultado puede no generalizar a condiciones reales. Por ello, una evaluación rigurosa debería explicitar el origen de los datos y evitar presentar resultados obtenidos exclusivamente en simulación como evidencia concluyente de desempeño en un entorno operativo.

## 4.3. Tabla comparativa de conjuntos de datos públicos

Los conjuntos siguientes fueron seleccionados por su relación directa con los problemas del proyecto o por servir como referencia metodológica. **«Sí» en la columna de equipos significa que se pueden analizar grabaciones existentes sin disponer de hardware RF; no significa que las capturas originales se hayan obtenido sin equipos.**

| Conjunto y fuente | Datos / procedencia | Etiquetas | Aplicación al proyecto | Acceso y tamaño publicado | ¿Sin equipo RF? | Principal limitación |
|---|---|---|---|---|---|---|
| **LOG-a-TEC `ws_traffic_20170606`** [2] | Mediciones reales de PSD en 868 MHz, durante 24 h; 5 medidas/s, 1024 *bins*, ancho de 192 kHz; JSONL | **Sí**, etiquetas de transmisiones elaboradas por dos expertos, con variantes según umbral | Detectar transmisiones y estimar ocupación local en tiempo/frecuencia | Descarga pública; tamaño del ZIP no indicado en la ficha consultada; licencia **CC BY-NC-SA 4.0** | **Sí** | Una única banda estrecha y un entorno concreto; las etiquetas dependen de criterios de anotación |
| **RF Spectrum-Sharing Challenge (RFSSC, GENESYS)** [3] | Escenarios **simulados** de señales radar/comunicaciones; muestras I/Q en SigMF a 50 MHz | **Sí**, matrices de ocupación por clase de usuario y matrices de posibilidad de transmisión para escenas de ejemplo y prueba | Ocupación, espacios en blanco sujetos a reglas y solapamiento entre usuarios | Enlace público a Dataverse desde GENESYS; **acceso al contenido y tamaño total no comprobados** en esta revisión | **Sí** | Escenarios simulados, archivos potencialmente pesados y reglas específicas del desafío |
| **AERPAW Packapalooza 2025** [4] | Mediciones **reales** de potencia y registros GPS durante unas 6 h de barridos urbanos hasta 6 GHz; SigMF | **No** se publican etiquetas ocupado/libre por intervalo en la ficha | Mapas de ocupación y exploración de bandas aparentemente libres | Descarga pública en Dryad; **438,56 MB** | **Sí** | Requiere definir/validar criterios de ocupación; recorrido y duración limitados |
| **AERPAW, nodos fijos CC1, CC2 y LW1 (datos de feb. 2022; publicación 2026)** [5] | Mediciones **reales** entre ~87 MHz y 6 GHz, tres sensores, una semana; potencia por frecuencia y tiempo; SigMF | **No** se describen etiquetas binarias de ocupación | Predicción temporal, comparación urbano/rural y ocupación espectral | Descarga pública en Dryad; **52,25 GB** en total, repartidos en tres ZIP | **Sí** | Requiere mucho almacenamiento; datos sin *ground truth* directo de ocupación |
| **GNSS Spectrum & Low-Cost Laboratory Dataset (Fraunhofer)** [6] | Capturas **reales** de I/Q y características de receptor GNSS en escenarios controlados | **Sí**, nueve clases en el subconjunto FIOT: sin interferencia y ocho tipos de interferencia; particiones de entrenamiento/prueba | Detección y clasificación de interferencias | Zenodo ofrece un ZIP de **221,8 kB** con instantánea y punteros; el proyecto declara ~**700 MB**, cifra que **no equivale al tamaño del ZIP ni se verificó descargando los binarios**. Requiere Git LFS | **Sí** | Centrado en GNSS; descarga y reconstrucción menos directas de lo que sugiere el tamaño publicado |
| **Indoor Jamming Scenarios (Alhazbi y colaboradores)** [7] | Muestras I/Q **reales** de comunicaciones BPSK en oficina, con diversos dispositivos, distancias y potencias de interferencia | **Sí**, tres condiciones (sin jamming, tono y ruido gaussiano) identificadas en las matrices de los archivos `.mat` | Detección y tipificación de interferencias | Zenodo lista **118,5 GB** en **31 archivos `.mat`** (más README), con archivos individuales de varios GB | **Sí** | Elevado volumen; escenarios indoor y modulación de interés limitada a BPSK |
| **Extended Spectrogram Dataset (Fraunhofer, 2026)** [8] | **30.000 espectrogramas sintéticos**, transmisiones Wi-Fi, Bluetooth y 5G, incluyendo coexistencia | **Sí**, el repositorio declara espectrogramas anotados para detección/localización de RF; el **esquema concreto de anotaciones no se comprobó en los archivos** | Localización de transmisiones y estudio de coexistencia/solapamiento | Descarga en Fordatis; ZIP de **19,61 GB** | **Sí** | Sintético, demanda almacenamiento y no equivale a etiquetas de interferencia perjudicial |
| **NASCTN CBRS SEA Data (NIST)** [9] | Mediciones **reales** de potencia por frecuencia y tiempo en ~3530–3710 MHz (2024–2026), en varios sensores | **No** se describen etiquetas ML universales de ocupación o interferencia | Ocupación de banda compartida y series temporales reales | Descarga pública sin restricciones por fecha/sensor en NIST; alternativa Box requiere invitación; tamaño total no indicado | **Sí** | Cobertura restringida a CBRS y necesidad de construir etiquetas/seleccionar subconjuntos |
| **RadioML 2016.10A (DeepSig)** [10] | Muestras I/Q **sintéticas** de 11 modulaciones y diferentes relaciones señal/ruido | **Sí**, clase de modulación y SNR | Referencia metodológica secundaria para trabajar con I/Q | Descarga histórica pública; formato Python *pickle* | **Sí** | **No** contiene por definición etiquetas de interferencia ni mapas de ocupación: **no debería ser el conjunto principal del proyecto** |

## 4.4. Análisis de los conjuntos más relevantes

### 4.4.1. LOG-a-TEC: punto de partida para estudiar ocupación y detección de transmisiones

Este conjunto procede de una campaña de 24 horas en la banda europea de 868 MHz. Se registraron aproximadamente cinco mediciones de PSD por segundo y cada observación contiene 1024 valores de potencia distribuidos en un ancho de banda de 192 kHz. Los archivos se entregan en formato **JSONL**, con campos de tiempo y mediciones. El repositorio ofrece además **anotaciones de transmisiones realizadas por dos expertos**, con distintas configuraciones de umbral [2].

Su principal ventaja es que vincula **mediciones reales** y **etiquetas externas**, lo que permite comparar una detección automática de transmisiones con anotaciones existentes. Puede utilizarse para estudiar periodos de actividad y silencio dentro de la banda medida. Su limitación consiste en que no representa el comportamiento de todo el espectro ni ofrece automáticamente etiquetas de «interferencia perjudicial» o «frecuencia legalmente disponible».

**Valoración:** **prioridad alta** para un prototipo universitario de detección/ocupación, por su formato relativamente accesible y sus etiquetas. Antes de descargarlo, debe comprobarse el tamaño real del archivo y la estructura de los archivos de anotación.

### 4.4.2. RFSSC: verdad terreno explícita para ocupación y acceso al espectro

El *First Radio-Frequency Spectrum-Sharing Challenge* proporciona escenas sintéticas con señales radar y de comunicaciones; GENESYS aclara explícitamente que **su primera edición utilizó únicamente señales simuladas**. Los escenarios se muestrean a **50 MHz** y la documentación describe **tres matrices binarias de ocupación** según el tipo de usuario y **dos matrices de posibilidad de transmitir**, calculadas conforme a reglas de prioridad. El repositorio indica que se publicaron matrices de verdad terreno para **tres escenas de ejemplo y seis de prueba**, junto con señales y escenas [3].

Este conjunto es especialmente pertinente porque permite diferenciar dos preguntas que suelen confundirse:

1. **¿Hay alguna señal presente en esta celda tiempo-frecuencia?** — ocupación.
2. **¿Puede transmitir determinado usuario en esta celda?** — posibilidad de acceso bajo las reglas definidas.

Por tanto, resulta más apropiado que un dataset de clasificación de modulación para evaluar directamente un **mapa de ocupación**. No obstante, sus reglas son las de un desafío experimental y sus escenas son sintéticas; no equivalen a condiciones regulatorias universales. GENESYS publica el enlace a Dataverse, pero **no se comprobó la lectura o descarga de los ficheros individuales**, por lo que su disponibilidad efectiva para el equipo sigue pendiente de prueba.

**Valoración:** **prioridad alta** como referencia de evaluación con verdad terreno; comenzar por las escenas de ejemplo y las matrices asociadas, no por todo el desafío.

### 4.4.3. AERPAW: mediciones reales para investigar espacios en blanco

AERPAW publica campañas de monitoreo con barridos de frecuencia, valores de potencia y registros temporales. Para un equipo con recursos limitados, destaca **Packapalooza 2025** (438,56 MB): mediciones de un entorno urbano recogidas durante aproximadamente seis horas, con barridos de hasta 6 GHz y registros GPS; los archivos de potencia pueden convertirse a MATLAB y su documentación identifica variables `freqs` (MHz) y `powers` (dBm) [4].

El conjunto **de nodos fijos**, publicado en marzo de 2026, aporta una escala temporal mayor: **una semana**, tres puntos de medición y 98.868 intervalos de frecuencia por barrido, de ~87 MHz a 6 GHz. Está distribuido en tres ZIP que suman **52,25 GB** e incluye scripts de ejemplo en Python/MATLAB [5].

Estos materiales permiten explorar patrones de actividad y estimar ocupación a partir de criterios explícitos, por ejemplo un umbral relativo al piso de ruido. **Las etiquetas ocupado/libre que se generen de este modo serán inferidas**, no verdad terreno provista por los autores. Es imprescindible registrar el método usado y reconocer el riesgo de falsos espacios libres por señales débiles, desvanecimiento, condiciones del receptor o el problema del nodo oculto.

**Valoración:** comenzar por **Packapalooza 2025**; reservar el conjunto de **52,25 GB** para una fase posterior de análisis temporal y generalización.

### 4.4.4. Fraunhofer GNSS: clasificación supervisada de interferencias

La colección *Spectrum & Low-Cost (LC) Laboratory Dataset for Interference Classification in GNSS Data* reúne capturas de laboratorio obtenidas con hardware especializado. La documentación describe instantáneas I/Q y etiquetas del subconjunto FIOT en **nueve clases**: una clase sin interferencia y ocho grupos de interferencias, entre ellas ruido, *chirp*, salto de frecuencia, señales moduladas, multitono, pulsos y tonos [6]. También se documentan particiones de entrenamiento y prueba.

Esta organización puede servir directamente para un problema supervisado **sin interferencia / con interferencia** o **clasificación multiclase**. Sin embargo, no debe ocultarse una dificultad práctica: la ficha de Zenodo explica que el ZIP contiene sobre todo un registro del repositorio y **punteros Git LFS**. Para obtener los binarios completos hay que descargar los objetos reales mediante Git LFS desde el repositorio de proyecto vinculado y extraer los archivos `.7z` según el subconjunto. **Atención:** el propio texto de Zenodo combina referencias a `FIOT_LC_laboratory` con instrucciones para otro repositorio (`FraunhoferIIS_Jammertest2025`); por eso, antes de ejecutar comandos, hay que contrastar la ruta con el repositorio enlazado a este dataset y verificar qué archivos contiene. Las instrucciones sobre archivos HDF5 fragmentados no deben asumirse aplicables a este subconjunto sin comprobarlo. Por tanto, **no se debe presentar el ZIP de Zenodo como descarga autosuficiente de todos los datos**.

**Valoración:** **prioridad media-alta** para la línea de interferencias, siempre que el equipo compruebe primero el acceso efectivo a los binarios y el tamaño final de la muestra seleccionada. **El archivo descargable directamente de Zenodo es de solo 221,8 kB; no debe confundirse con la colección de señales.** Los resultados no deberían extrapolarse sin más a todas las bandas, pues el contexto es GNSS.

### 4.4.5. Indoor Jamming Scenarios: datos RF reales con clases de interferencia

Alhazbi, Sciancalepore y Oligeri publicaron capturas I/Q reales obtenidas en escenarios de comunicación BPSK con un interferente. La ficha diferencia condiciones sin jamming, con tono y con ruido gaussiano, además de variaciones de equipo, distancia y potencia. Los archivos están disponibles individualmente en `.mat`; el artículo original confirma **31 archivos**, cada uno con matrices `Nojamming`, `Sine` y `Gaussian` [7].

Su aportación al estado del arte reside en la posibilidad de estudiar señales **reales** afectadas por interferencias bajo distintas condiciones de medición. Como inconveniente, el tamaño publicado es **118,5 GB**: no es razonable descargarlo íntegramente como primera tarea de un proyecto universitario. Puede seleccionarse un subconjunto de condiciones y documentar qué escenarios quedan representados.

**Valoración:** fuente científicamente pertinente, pero **prioridad operativa media-baja** por el volumen de datos.

### 4.4.6. Fraunhofer Extended Spectrogram: imágenes y coexistencia de tecnologías

El repositorio de Fraunhofer, publicado en septiembre de 2026, describe **30.000 espectrogramas sintéticos** destinados a detectar/localizar tramas RF de **Wi-Fi, Bluetooth y 5G**, extendiendo un trabajo anterior que ya incluía colisiones entre Wi-Fi y Bluetooth. La descarga publicada tiene **19,61 GB** [8].

Es interesante cuando la pregunta de investigación se formula como **localización de transmisiones sobre una representación tiempo-frecuencia**. No obstante, detectar que dos transmisiones coexisten no demuestra por sí mismo que exista una interferencia perjudicial: sería necesario definir la señal de interés y una condición de degradación, o presentar el problema con precisión como detección de actividad/solapamiento.

**Valoración:** recurso complementario; no es la primera opción si el equipo dispone de poco almacenamiento.

### 4.4.7. NIST NASCTN CBRS SEA: una referencia institucional para ocupación de espectro compartido

El programa **NASCTN CBRS SEA**, coordinado por NIST y otras instituciones, publica mediciones reales de potencia selectivas en tiempo y frecuencia, en aproximadamente **3530–3710 MHz**, recogidas entre el verano de **2024 y el verano de 2026**. **El intervalo medido es más amplio que la banda regulatoria CBRS de 3550–3700 MHz.** El acceso se ofrece por sensor y fecha mediante el repositorio público de NIST; también se describen formatos SigMF y herramientas para visualizar subconjuntos [9].

Su valor consiste en disponer de una fuente institucional para investigar cómo cambia la actividad en una banda de uso compartido. Para convertirla en una tarea de aprendizaje supervisado de ocupación, el equipo tendría que construir y justificar las etiquetas o seleccionar un protocolo de evaluación que no dependa de atribuir a las mediciones una verdad terreno inexistente.

**Valoración:** fuente sólida para ampliar el análisis de ocupación; preferible seleccionar un intervalo de tiempo y un sensor antes que descargar todo el archivo histórico.

### 4.4.8. Por qué RadioML no es suficiente

RadioML 2016.10A se emplea ampliamente para clasificación de modulaciones, con señales sintéticas I/Q de once clases y diferentes relaciones señal/ruido. Puede servir para aprender a manipular datos I/Q, pero **sus etiquetas no representan canales disponibles ni tipos de interferencia**. Además, el propio proveedor, DeepSig, advierte que sus colecciones académicas históricas presentan problemas conocidos y recomienda cautela y datos reales para nuevas investigaciones [10].

**Valoración:** utilizarlo solo como apoyo metodológico, no como evidencia principal para las dos preguntas del proyecto.

## 4.5. Requisitos de acceso, almacenamiento y procesamiento

### 4.5.1. Acceso y permisos

La mayoría de los repositorios considerados permiten **consultar y descargar datos ya capturados** sin disponer de un receptor SDR. Esto evita el coste de adquisición de hardware, pero **no elimina** la necesidad de conexión, espacio de disco, memoria y herramientas de lectura. Deben verificarse las licencias de reutilización y las condiciones de cada repositorio antes de redistribuir los archivos o publicarlos junto con un modelo. En particular, LOG-a-TEC y RadioML indican licencia **CC BY-NC-SA 4.0** [2, 10].

### 4.5.2. Formatos y herramientas

| Formato | Datasets de ejemplo | Herramientas posibles sin hardware RF |
|---|---|---|
| `JSONL` | LOG-a-TEC | Python, `json`, `pandas` |
| `.sigmf-data` + `.sigmf-meta` | AERPAW, RFSSC, NIST | Python (`numpy`, lector SigMF), scripts de conversión del repositorio |
| `.mat` | Indoor Jamming, matrices RFSSC y conversiones AERPAW | MATLAB, Python (`scipy.io`; `h5py` para determinados MAT v7.3) |
| `.npy`, HDF5 y etiquetas de texto | Fraunhofer GNSS (según los subconjuntos) | Python (`numpy`, `h5py`), Git/Git LFS cuando corresponda |
| Imágenes/espectrogramas y anotaciones | Fraunhofer Extended Spectrogram | Python, bibliotecas para imágenes y lectura del formato de anotación |
| *Pickle* | RadioML 2016.10A | Python; **no cargar archivos pickle de fuentes no confiables**, ya que su deserialización puede ejecutar código |

**Advertencia:** los datos de radio y la documentación de adquisición deben leerse conjuntamente. En SigMF, el tipo de dato lo determina el metadato de cada conjunto. No debe emplearse el mismo lector para archivos de **potencia `float32`** y para grabaciones **I/Q complejas `cf32`** sin comprobar previamente el formato.

### 4.5.3. Viabilidad con recursos limitados

- **Inicio con recursos modestos:** LOG-a-TEC y AERPAW Packapalooza 2025. Permiten practicar análisis de potencia y actividad, con descargas publicadas relativamente manejables para el segundo conjunto.
- **Interferencias etiquetadas:** Fraunhofer GNSS, siempre que el acceso vía Git LFS se confirme; de lo contrario, probar una selección pequeña de archivos de *Indoor Jamming Scenarios*.
- **Verdad terreno clara de ocupación:** RFSSC, trabajando primero con matrices y escenas de ejemplo.
- **Ampliación posterior:** AERPAW de nodos fijos, Fraunhofer de espectrogramas y NIST CBRS, procesando por partes y sin cargar todas las mediciones simultáneamente en memoria.

El espacio necesario puede ser **mayor que el tamaño descargado**, debido a archivos comprimidos, transformaciones, espectrogramas intermedios y particiones de entrenamiento. Se recomienda procesar los archivos en fragmentos y preservar una copia de los metadatos, parámetros y etiquetas originales.

## 4.6. Etiquetado y calidad de los datos: consideraciones para el proyecto

Las decisiones de etiquetado afectan directamente la validez del experimento.

1. **Etiquetas explícitas frente a etiquetas derivadas.** Para interferencias, un archivo con condiciones experimentales conocidas o clases anotadas permite evaluar un modelo frente a categorías predefinidas. Para AERPAW o NIST, la etiqueta ocupado/libre probablemente deba inferirse de las mediciones: es una **etiqueta operativa**, no una observación independiente de la verdad.
2. **Umbrales y piso de ruido.** Si se asigna `ocupado = 1` cuando la potencia supera un umbral, el resultado dependerá de la calibración, del ancho de banda de resolución y del ruido. Es conveniente documentar un análisis de sensibilidad del umbral.
3. **Fuga de información (*data leakage*).** Ventanas consecutivas de la misma captura pueden ser muy parecidas. Dividirlas aleatoriamente entre entrenamiento y prueba puede producir resultados artificialmente favorables. Para campañas reales, conviene reservar periodos completos o sensores/entornos diferentes para la evaluación, si los datos lo permiten.
4. **Desbalance de clases.** Puede haber más observaciones de actividad normal que de interferencia, o viceversa. Antes de interpretar una métrica global, debe inspeccionarse la distribución de clases.
5. **Generalización.** Un detector probado en GNSS, en un solo edificio o con un único transmisor no queda validado automáticamente para otras bandas, equipos y ubicaciones.
6. **Disponibilidad no equivale a permiso.** Una celda sin energía detectable no garantiza que sea seguro o legal transmitir en ella; es una limitación fundamental del problema de espacios en blanco.

Estas cuestiones deben compartirse con las secciones del equipo dedicadas a preparación de datos (Diego), detección de interferencias (Daniela), espacios en blanco (Fabián) y evaluación (Rubén), sin sustituir sus análisis específicos.

## 4.7. Recomendación de selección para una implementación académica

La revisión identifica una ruta viable **sin adquisición de equipos RF** y sin decidir prematuramente un algoritmo:

**Ruta A — Identificación de ocupación y espacios en blanco.** Emplear inicialmente **LOG-a-TEC**, para contrastar detecciones con sus etiquetas de transmisiones, o **AERPAW Packapalooza 2025**, para estudiar mediciones reales de frecuencia y potencia. El resultado mínimo sería una matriz de actividad por tiempo/frecuencia y una definición explícita de «canal aparentemente libre». Si se necesita una referencia de ocupación más objetiva y controlada, incorporar las matrices de **RFSSC**, distinguiendo la ocupación de la posibilidad de transmitir.

**Ruta B — Detección de interferencias.** Seleccionar **Fraunhofer GNSS** porque distingue explícitamente una clase sin interferencia y varias clases de interferentes. La primera tarea técnica no es entrenar un modelo, sino **verificar la obtención de un subconjunto real de binarios mediante Git LFS**. Si ese acceso se complica, usar una muestra documentada de *Indoor Jamming Scenarios*, cuidando el tamaño por archivo.

**Recomendación de integración:** utilizar **dos subconjuntos distintos**, uno para ocupación y otro para interferencias. **Entre los nueve recursos examinados no se identificó** uno pequeño, real, público y con etiquetas completas de referencia para ambas tareas a la vez; esto no demuestra que no exista en la literatura. La integración conceptual puede realizarse después sobre representaciones compatibles —por ejemplo, ventanas tiempo-frecuencia— sin afirmar que las etiquetas tienen el mismo significado.

### Comprobaciones iniciales sugeridas

| Paso | Verificación | Resultado mínimo esperado |
|---|---|---|
| 1 | Abrir las páginas de licencia y descarga de los dos datasets elegidos | Fuente, versión, fecha y condiciones de uso registradas |
| 2 | Descargar **un** archivo de ejemplo o subconjunto pequeño | Lectura reproducible en Python |
| 3 | Comprobar forma, unidades y metadatos | Tabla breve de número de registros, frecuencias, tiempos y etiquetas disponibles |
| 4 | Visualizar una muestra | Curva de potencia, espectrograma o mapa de ocupación, según el dato |
| 5 | Documentar cómo se define la etiqueta | Diferenciar etiqueta de autor y etiqueta inferida por el equipo |
| 6 | Fijar una partición de evaluación sin fuga temporal | Conjunto de entrenamiento y prueba con separación justificada |

## 4.8. Respuestas a las preguntas de investigación asignadas

**1. ¿Qué conjuntos de datos públicos existen para detectar interferencias?**  
Fraunhofer GNSS, *Indoor Jamming Scenarios* y el conjunto de espectrogramas de Fraunhofer ofrecen, respectivamente, clases explícitas de interferencia, capturas reales con condiciones de jamming identificadas y escenas anotadas de transmisiones/coexistencia. Los dos primeros son más directos para clasificar interferencias; el tercero requiere no confundir coexistencia con degradación.

**2. ¿Qué conjuntos permiten identificar espacios en blanco u ocupación?**  
LOG-a-TEC, AERPAW (campañas urbana y fija), RFSSC y NIST CBRS. RFSSC proporciona mapas de verdad terreno en escenas simuladas; los conjuntos de medición real requieren evaluar la actividad y, en muchos casos, construir una etiqueta de ocupación basada en criterios explícitos.

**3. ¿Qué contiene cada conjunto?**  
Las fuentes ofrecen distintas representaciones: mediciones de potencia con tiempo y frecuencia, I/Q, espectrogramas, etiquetas de transmisiones, clases de interferencia, mapas binarios de ocupación y metadatos. Su contenido específico figura en la tabla comparativa y en las secciones 4.4 y 4.5.

**4. ¿Los datos están etiquetados?**  
No todos. LOG-a-TEC, RFSSC, Fraunhofer GNSS, *Indoor Jamming Scenarios* y el conjunto de espectrogramas incluyen alguna forma de anotación o condición experimental conocida. AERPAW y NIST no presentan en las fichas examinadas un conjunto universal de etiquetas ocupado/libre listo para entrenamiento supervisado.

**5. ¿Se pueden utilizar sin equipos de radio?**  
Sí: los datos ya fueron adquiridos o simulados por sus productores. Para analizarlos no se necesita comprar un SDR, aunque las descargas, conversiones y experimentos sí requieren ordenador, software y recursos de almacenamiento.

**6. ¿Qué requisitos tienen?**  
Dependen del conjunto. Hay archivos JSONL manejables con Python, repositorios de algunos cientos de MB, archivos `.mat` de varios GB y colecciones que superan las decenas de GB. Fraunhofer GNSS tiene una particularidad de acceso mediante Git LFS. También deben considerarse formatos, metadatos, licencias, calidad de las etiquetas y memoria disponible.

## 4.9. Conclusiones

La revisión muestra que **el principal obstáculo para un proyecto académico no es la inexistencia de datos abiertos, sino la correspondencia entre los datos y la pregunta de investigación**. Una colección de modulaciones no equivale a un conjunto de interferencias, un espectrograma con transmisiones simultáneas no prueba interferencia perjudicial y una medición de potencia no determina por sí misma si una frecuencia está autorizada para uso secundario.

Para un desarrollo inicial sin hardware especializado, **LOG-a-TEC y AERPAW Packapalooza 2025** son candidatos relevantes para estudiar ocupación y actividad real; **RFSSC** complementa el análisis con mapas de ocupación y reglas de acceso explícitas, mientras que **Fraunhofer GNSS** aporta etiquetas de interferencia que facilitan una evaluación supervisada. La elección final debe basarse en una pequeña prueba de descarga y lectura, no únicamente en la descripción del repositorio.

En consecuencia, la recomendación es construir **dos líneas experimentales delimitadas**, con datos, etiquetas y criterios de validación documentados, antes de intentar una solución unificada de monitoreo y gestión del espectro.

## 4.10. Fuentes y referencias consultadas

1. **SigMF.** *Signal Metadata Format Specification*. https://sigmf.org/
2. **SensorLab / LOG-a-TEC.** *LPWAN Trace Sets: `ws_traffic_20170606`*. Incluye enlace al trabajo de Gale et al., “Automatic Detection of Wireless Transmissions”, *IEEE Access* (2020). https://log-a-tec.eu/datasets.html
3. **GENESYS.** *Dataset for the First Radio-Frequency Spectrum-Sharing Challenge (RFSSC)*. Descripción de escenarios, señales, mapas de ocupación y enlaces a materiales en Dataverse. https://genesys-lab.org/radio-frequency-spectrum-sharing
4. **Raouf, A. H., Ozdemir, O., Guvenc, I. y Sichitiu, M.** (2025, datos publicados en Dryad). *AERPAW helikite spectrum measurements at Packapalooza Festival in Aug 2025*. https://doi.org/10.5061/dryad.tb2rbp0fw
5. **Raouf, A. H., Maeng, S. J., Ozdemir, O., Guvenc, I. y Sichitiu, M.** (2026). *AERPAW sub-6 GHz spectrum monitoring dataset: Fixed nodes CC1, CC2, LW1 (February 2022)*. Dryad. https://doi.org/10.5061/dryad.hmgqnk9zn
6. **Brieger, T. y colaboradores.** (registro Zenodo publicado el 13 de julio de 2026). *Spectrum & Low-Cost (LC) Laboratory Dataset for Interference Classification in GNSS Data*. Zenodo, descarga archivada de **221,8 kB** y datos grandes enlazados por Git LFS. https://zenodo.org/records/21333335; repositorio de proyecto: https://github.com/FelixOtt94/FIOT_LC_laboratory. Véase también su artículo asociado: *Multimodal Learning for Reliable Interference Classification in GNSS Signals* (ION GNSS+, 2022), https://doi.org/10.33012/2022.18586
7. **Alhazbi, S., Sciancalepore, S. y Oligeri, G.** (2022). *A Dataset of IQ samples in Indoor Jamming Scenarios*. Zenodo. https://doi.org/10.5281/zenodo.7119040. Artículo de datos: *A Dataset of Physical-Layer Measurements in Indoor Wireless Jamming Scenarios*, *Data in Brief*, **vol. 46 (febrero de 2023), art. 108773; publicado en línea el 23 de noviembre de 2022**. https://doi.org/10.1016/j.dib.2022.108773
8. **Akshaya Bindu Gowri, J. Wicht y H. Salehi.** (2026). *Extended Spectrogram Dataset for Deep Learning-Based RF-Frame Detection: Wi-Fi, Bluetooth, and 5G Packets*. Fraunhofer Fordatis. https://doi.org/10.24406/fordatis/478
9. **NIST / NASCTN.** *CBRS Sharing Ecosystem Assessment (SEA): Data Access*. Datos de monitoreo publicados para 2024–2026. https://pages.nist.gov/SEA-DATA/
10. **DeepSig.** *Datasets: Historical RadioML collections and usage cautions*. https://www.deepsig.ai/datasets/


## 4.11. Auditoría de verificabilidad de los datos y de las referencias

Esta subsección registra **qué se verificó en las fuentes primarias** y qué aspectos requieren una inspección de los archivos antes de comenzar la implementación. «Verificado» aquí significa **confirmado documentalmente** en el registro del proveedor, no que las grabaciones hayan sido descargadas y validadas.

| Dataset | ¿El repositorio y los metadatos principales existen? | Naturaleza de los datos | Descarga/lectura de binarios comprobada | Advertencia de uso |
|---|---|---|---|---|
| LOG-a-TEC | Sí; ficha oficial de SensorLab y artículo IEEE Access | Captura real | **No** | Revisar ZIP, anotaciones de los dos expertos y cobertura de solo 192 kHz |
| RFSSC / GENESYS | Sí; descripción técnica oficial, matrices de ocupación y enlace a Dataverse | **Simulación** | **No** | La participación en el desafío y el conjunto de prueba no sustituyen un entrenamiento con mediciones reales; el enlace de Dataverse no se abrió satisfactoriamente en la verificación |
| AERPAW Packapalooza 2025 | Sí; Dryad confirma fecha, duración, 438,56 MB y archivos SigMF | Captura real | **No** | No hay etiquetas de canal libre verificadas como verdad terreno independiente |
| AERPAW nodos fijos | Sí; Dryad confirma tres sensores, 52,25 GB, una semana y ~98.868 bins por barrido | Captura real | **No** | Descargar por partes; confirmar ocupación calculada y evitar fuga temporal |
| Fraunhofer GNSS FIOT/LC | Sí; Zenodo, GitHub y publicación ION GNSS+ | Captura real en laboratorio controlado | **No** | ZIP de Zenodo **221,8 kB**; los binarios requieren Git LFS y comprobar rutas; **700 MB** es tamaño declarado del proyecto, no tamaño comprobado de señales descargadas |
| Indoor Jamming | Sí; Zenodo y artículo en *Data in Brief* | Captura real en laboratorio | **No** | **118,5 GB**; tres matrices de condiciones por archivo; la correspondencia etiqueta-escenario debe comprobarse al leer una muestra |
| Fraunhofer Extended Spectrogram | Sí; Fordatis confirma 30.000 ejemplos y ZIP de 19,61 GB | **Simulación** | **No** | Las anotaciones detalladas se deben inspeccionar; coexistencia no es sinónimo de interferencia perjudicial |
| NIST NASCTN CBRS SEA | Sí; NIST confirma 3530–3710 MHz, verano 2024–verano 2026 y descargas por sensor/día | Captura real | **No** | No se identificó una etiqueta universal de ocupación independiente en la página; descarga pública por NIST, Box con invitación |
| RadioML 2016.10A | Sí; página de DeepSig | **Simulación** | **No** | Etiquetas de modulación/SNR, **no** de jamming ni de ocupación; erratas reconocidas por el proveedor |

### Correcciones documentales efectuadas durante la revisión

1. **Naturaleza de las observaciones.** Se diferenció la existencia verificable de una fuente frente a la procedencia real o simulada de las señales. Los **9 repositorios son identificables**; **6 contienen capturas reales** y **3 contienen datos sintéticos**.
2. **Fraunhofer GNSS: tamaño y vía de acceso.** Zenodo lista un ZIP de **221,8 kB**, mientras el README indica **700 MB** como tamaño de proyecto. Los datos de gran tamaño dependen de Git LFS; **no se verificó que se puedan recuperar todos los objetos**. Las instrucciones mezcladas de distintos repositorios deben comprobarse antes de ejecutarlas.
3. **Indoor Jamming: año bibliográfico.** La captura y el depósito son de 2022, pero el artículo de *Data in Brief* pertenece formalmente al **volumen 46 de febrero de 2023**, tras publicarse en línea en noviembre de 2022. Se añadió el DOI correcto del artículo.
4. **RFSSC: alcance y disponibilidad.** GENESYS define el experimento como un sistema de compartición ficticio y describe señales **solo simuladas**; el enlace a Dataverse existe, pero **no se comprobó la descarga** de sus archivos.
5. **Las etiquetas no son intercambiables.** «Transmisión detectada», «ocupación medida», «permiso de transmitir» e «interferencia perjudicial» son objetivos y referencias distintos. No se afirmarán resultados experimentales con los conjuntos hasta comprobar su contenido y protocolo de validación.

**Conclusión de la auditoría:** las referencias principales y los números citados cuentan con respaldo documental, con las precisiones anteriores. **No sería correcto afirmar que los nueve conjuntos contienen mediciones reales ni que se ha validado experimentalmente su integridad.** Antes de justificar una elección definitiva de dataset, hay que descargar y abrir una muestra, identificar las etiquetas efectivas y documentar sus licencias y el tamaño real de trabajo.

---

*Alcance de esta parte: disponibilidad, características, etiquetas y viabilidad de los datos. La elección de algoritmos, el preprocesamiento detallado y la evaluación de modelos corresponden a las otras secciones del trabajo grupal.*
