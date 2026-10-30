# Estado del arte: detección de interferencias en el espectro radioeléctrico mediante Machine Learning

---

## 1. Introducción

El espectro radioeléctrico es un recurso escaso y regulado: cada banda está asignada a servicios concretos (radiodifusión, telefonía móvil, navegación por satélite, radioastronomía). Cuando una emisión no autorizada, fuera de banda o mal configurada ocupa esas frecuencias, se produce una **interferencia** que degrada o interrumpe el servicio afectado. Por ello, reguladores y operadores monitorean el espectro de forma continua para detectar, localizar y clasificar interferencias.

La detección tradicional se apoya en umbrales de potencia y en la inspección manual de expertos, un enfoque que no escala frente al crecimiento del número de emisores (5G, IoT, drones, enlaces satelitales) ni frente al volumen de datos de los receptores modernos. El aprendizaje automático (Machine Learning, ML) permite que los modelos aprendan las firmas de las interferencias a partir de las mediciones y las detecten de forma automática.

Esta revisión compara las técnicas de ML aplicadas a la **detección de interferencias**: qué datos emplean, qué algoritmos utilizan, cómo se evalúan y qué limitaciones presentan. Responde a cinco preguntas de investigación (sección 10) y cierra con recomendaciones prácticas (sección 11).

## 2. Alcance y método de búsqueda

La revisión se limita a la detección de interferencias. No profundiza en la identificación de espacios en blanco ni en la predicción de ocupación espectral, ni en la preparación detallada de datos o la implementación práctica, aunque se mencionan cuando son necesarios para entender los resultados.

Los artículos se localizaron en Google Scholar, arXiv, IEEE Xplore y ScienceDirect, con términos como *"radio frequency interference detection"*, *"machine learning"*, *"RFI flagging deep learning"*, *"GNSS jamming identification"* y *"spectrum anomaly detection autoencoder"*. Se priorizaron investigaciones originales que reportan datos, algoritmos y métricas verificables.

El dominio de aplicación resultó un criterio útil de organización, porque el problema cambia según el escenario:

- **Radioastronomía:** las interferencias contaminan las observaciones y deben "marcarse" (*RFI flagging*) antes de procesar los datos.
- **Navegación por satélite (GNSS):** se detectan interferencias intencionales como el *jamming* (bloqueo) y el *spoofing* (engaño).
- **Monitoreo del espectro y espectro compartido:** se buscan transmisiones no autorizadas en bandas reguladas.
- **Teledetección pasiva:** los radiómetros científicos sufren interferencias de radares, 5G y drones en bandas teóricamente protegidas.

## 3. Tipos de interferencias estudiadas en la literatura

**Por estructura espectral y temporal.** Liu et al. (2025), en su estudio sobre señales de navegación, distinguen cinco tipos básicos: onda continua (*continuous wave*, CW), barrido de frecuencia (*chirp*), interferencia pulsada, de banda estrecha y de espectro adaptado (*matched-spectrum*). Además estudian las interferencias compuestas, que combinan varios tipos a la vez y siguen siendo el escenario más difícil de reconocer.

**Por intencionalidad.** Ghanbarzade y Soleimani (2025) separan el *jamming*, que satura el receptor con potencia elevada, del *spoofing*, que transmite señales falsas parecidas a las legítimas. En el monitoreo regulatorio la interferencia típica no es intencional: Tschimben y Gifford (2022) estudian transmisiones no autorizadas en espectro compartido, y Takács et al. (2026) analizan anomalías de emisoras FM como interrupciones de transmisión, pérdida de modulación o desplazamientos anómalos de frecuencia.

**Por sector de aplicación.** En radioastronomía el problema se conoce como RFI (*Radio Frequency Interference*) y afecta tanto a telescopios individuales como a interferómetros (Akeret et al., 2017; Vafaei Sadr et al., 2020; Pritchard et al., 2025). En teledetección pasiva, Alam (2024) documenta interferencias sobre radiómetros de microondas provenientes de radares de vigilancia aérea, comunicaciones 5G y drones.

Esta diversidad explica que no exista una única definición operativa de "interferencia": cada dominio define qué es anómalo según las señales legítimas que espera recibir.

## 4. Métodos tradicionales de detección: la línea base

Los métodos clásicos de detección de señales se basan en:

- **Energía recibida:** comparar la potencia medida contra un umbral.
- **Filtro adaptado:** correlacionar con una plantilla conocida de la señal.
- **Detección cicloestacionaria:** explotar las periodicidades estadísticas de las señales de comunicaciones.

Son útiles cuando la señal interferente es potente, conocida o estable, pero pierden eficacia con señales débiles, desconocidas o variables.

En radioastronomía, la línea base dominante es el método estadístico **SumThreshold** (Offringa et al., 2010), que marca como interferencia las muestras que superan umbrales adaptativos a lo largo de los ejes de tiempo y frecuencia. Los observatorios emplean variantes de estos algoritmos como *flaggers* por defecto, complementados con inspección manual. Así lo describen Akeret et al. (2017), que comparan su red neuronal contra SumThreshold, y Vafaei Sadr et al. (2020), que lo hacen contra el *flagger* por defecto del telescopio MeerKAT.

En teledetección pasiva, Alam (2024) señala el problema estructural de esta familia: son **algoritmos artesanales diseñados para un tipo concreto de interferencia**, por lo que cada nuevo patrón exige rediseñar el detector. A ello se suman el ajuste manual de umbrales, el elevado volumen de datos a inspeccionar y la sensibilidad a falsas alarmas cuando el umbral se baja para captar señales débiles. Estas carencias son el argumento que la literatura esgrime para introducir ML.

## 5. Técnicas de Machine Learning aplicadas a la detección de interferencias

### 5.1 Algoritmos clásicos supervisados

Los algoritmos clásicos (árboles de decisión, Random Forest, regresión logística, máquinas de vectores de soporte) aparecen sobre todo cuando se dispone de características numéricas extraídas de la señal y de ejemplos etiquetados de cada tipo de interferencia.

Ghanbarzade y Soleimani (2025) utilizaron Random Forest, árboles de decisión y regresión logística en sus experimentos iniciales de identificación de *jamming* y *spoofing* sobre dos conjuntos de datos reales. Su estudio completo, que combina ML, aprendizaje profundo y visión por computador, alcanzó una exactitud aproximada del **99 % en la detección de *jamming* de GNSS**, unos cinco puntos porcentuales por encima de los estudios previos que citan.

Estos métodos tienen ventajas prácticas con recursos limitados: necesitan relativamente pocas muestras, se entrenan rápido y sus resultados son más interpretables. Su contrapartida es que dependen de que alguien diseñe a mano las características de entrada, lo que reproduce el problema de "diseño artesanal" de los métodos clásicos.


### 5.2 Detección de anomalías: aprender qué es "normal"

Cuando no hay etiquetas de interferencias, la literatura recurre a métodos no supervisados que modelan el comportamiento normal del espectro y marcan como anómalo todo lo que se desvía de él. La familia más usada es la de los **autoencoders**: redes que aprenden a comprimir y reconstruir las mediciones normales, de modo que, ante una interferencia, la reconstrucción falla y el **error de reconstrucción** funciona como puntuación de anomalía.

Tschimben y Gifford (2022) aplicaron un autoencoder profundo y una variante con memoria temporal (LSTM-autoencoder) a muestras I/Q reales de un entorno de espectro compartido, e identificaron anomalías en el dominio del tiempo y de la frecuencia en forma de transmisiones no autorizadas.

Takács et al. (2026) realizaron un estudio sistemático sobre el conjunto de datos público de mediciones FM de Hungría (banda VHF II, 87,5–108 MHz). Compararon cinco variantes de autoencoder y un autoencoder variacional convolucional, con distintas estrategias de preprocesamiento y de puntuación del error (MSE, SSIM, estadísticos de dispersión). Su mejor configuración alcanzó un **F1 de 91,4 %** en el subconjunto de desarrollo. Dos conclusiones destacan: aumentar la complejidad de la arquitectura no mejoró necesariamente la detección, y la forma de puntuar el error de reconstrucción influyó de forma decisiva.

Métodos como Isolation Forest o one-class SVM pertenecen conceptualmente a esta misma línea de detección de anomalías. Sin embargo, en los estudios revisados no se encontraron aplicaciones con resultados propios de estos métodos a interferencias radioeléctricas; los autoencoders dominan, presumiblemente porque manejan bien datos de alta dimensionalidad como los espectrogramas.

### 5.3 Redes neuronales profundas: la tendencia dominante

El bloque más activo de la literatura trata la detección de interferencias como un problema de clasificación o segmentación de imágenes construidas a partir de la señal: espectrogramas tiempo-frecuencia, cascadas de agua (*waterfalls*) o visibilidades de interferómetro. Las redes neuronales convolucionales (CNN) dominan este espacio.

- **Akeret et al. (2017)** fueron de los primeros en aplicar una **U-Net** (CNN codificador-decodificador para segmentación) a datos 2D de tiempo-frecuencia de un radiotelescopio. Entrenaron y evaluaron con simulaciones del paquete HIDE & SEEK y con datos reales del telescopio de 7 m del Observatorio de Bleien, con una **precisión competitiva con SumThreshold**, y publicaron su software con licencia libre.
- **Vafaei Sadr et al. (2020)** propusieron **R-Net**, una arquitectura residual profunda que supera al *flagger* por defecto de MeerKAT y a las U-Net en AUC, F1 y MCC. Su precisión es aproximadamente un 90 % mayor que la del *flagger* con un recall del 80 %, y su F1 un 35 % superior. Su hallazgo metodológico más útil es el **aprendizaje por transferencia**: el modelo entrenado con simulaciones de MeerKAT y afinado con datos reales etiquetados a mano de KAT-7 alcanzó un **AUC de 0,91**, frente a 0,67 del mejor modelo sin transferencia.
- **Czech e Inggs (2018)** abordaron la clasificación de RFI transitorios combinando una **CNN con una LSTM**, para aprovechar la estructura espacial del espectrograma y la evolución temporal de la interferencia. Anticipan la idea de que el tipo de interferencia condiciona la arquitectura adecuada.
- **Liu et al. (2025)**, en GNSS, compararon una ResNet-18 con una red ligera propia (LcxNet-Fusion) sobre señales simuladas con cinco tipos de interferencia y sus combinaciones. Con entrada STFT alcanzaron una **sensibilidad de −20 dB de relación jamming/ruido (JNR) con 90 % de detección**, y de **−13 dB JNR con 90 % de identificación**. Su red redujo los parámetros un **43 %** (24,32 MB) y el tiempo de inferencia un **15 %** (1,25 s) respecto a ResNet-18.
- **Pritchard et al. (2025)** exploraron redes neuronales espiking (SNN): aplicaron *Liquid State Machines* con neuronas Leaky Integrate-and-Fire de segundo orden a datos simulados del interferómetro HERA, y alcanzaron una **exactitud por píxel del 98 % y un F1 de 0,743**. Argumentan que las CNN son costosas y de aplicabilidad limitada en tiempo real.
- **Supervisión débil:** Sharma et al. (2023) reportan exactitudes de prueba del 94,55 % y 93,69 % en astronomía sin etiquetado exhaustivo. También se han propuesto CNN compactas para detección y clasificación simultánea con exactitudes en torno al 94 % (Ayazbek et al.).
- **Alam (2024)** aplicó CNN a datos crudos de radiómetros de microondas (momentos y parámetros de Stokes), etiquetados dinámicamente mediante indicadores de calidad (*quality flags*), para detectar RFI de múltiples tipos a escala global sin diseñar un detector específico por tipo de interferencia.

### 5.4 Síntesis comparativa de las familias de técnicas

| Familia | Estudio | Datos | Resultado principal |
|---|---|---|---|
| Clásicos (RF, árboles, reg. logística) | Ghanbarzade y Soleimani, 2025 | 2 datasets reales GNSS | ~99 % de exactitud en *jamming* |
| Autoencoders (no supervisado) | Takács et al., 2026 | Dataset FM húngaro (Zenodo) | F1 = 91,4 % |
| Autoencoder / LSTM-AE | Tschimben y Gifford, 2022 | I/Q reales | Detecta transmisiones no autorizadas sin etiquetas |
| CNN (U-Net) | Akeret et al., 2017 | Simulado + real | Precisión competitiva con SumThreshold |
| CNN residual (R-Net) | Vafaei Sadr et al., 2020 | Simulado MeerKAT + real KAT-7 | AUC 0,91 con transferencia vs. 0,67 sin ella |
| CNN + LSTM | Czech e Inggs, 2018 | RFI transitorios | Los transitorios requieren memoria temporal |
| CNN ligera | Liu et al., 2025 | Simulado GNSS, 5 tipos | −20 dB JNR al 90 %; 43 % menos parámetros |
| Red espiking (LSM) | Pritchard et al., 2025 | Simulado HERA | 98 % exactitud por píxel; F1 = 0,743 |
| CNN con etiquetado débil | Alam, 2024 | Radiómetros (global) | Mejor generalización que detectores por tipo |

Puestas lado a lado, las tres familias presentan un compromiso claro entre precisión, requisitos de datos y coste:

- **Precisión:** las redes profundas registran los mayores rendimientos documentados (AUC de 0,91 con transferencia; ~99 % en *jamming* GNSS; 98 % por píxel; −20 dB JNR). Los autoencoders alcanzan valores sólidos pero algo inferiores (F1 de 91,4 %), y los clásicos rinden bien solo con características bien diseñadas.
- **Datos y etiquetado:** el orden se invierte. Los autoencoders son los menos exigentes (solo mediciones normales, sin etiquetas de interferencia); los clásicos necesitan pocas muestras etiquetadas y atributos elaborados a mano; las redes profundas son las más demandantes y suelen requerir afinación con datos reales.
- **Coste computacional:** los clásicos son ligeros e interpretables; los autoencoders tienen coste moderado; las CNN profundas son las más costosas, hasta el punto de motivar redes ligeras y arquitecturas espiking para viabilizar el tiempo real.

No existe una familia superior en todos los frentes: la elección depende de la abundancia de etiquetas, los datos disponibles y el presupuesto de cómputo.

## 6. Datos empleados por los estudios

**Tipo de dato de entrada**, en orden creciente de procesamiento:

- Muestras I/Q crudas (Tschimben y Gifford, 2022).
- Potencia por frecuencia y cascadas de agua (Takács et al., 2026).
- Momentos y parámetros de Stokes de radiómetro (Alam, 2024).
- Espectrogramas tiempo-frecuencia por STFT (Liu et al., 2025) o visibilidades de interferómetro (Vafaei Sadr et al., 2020; Pritchard et al., 2025).

Liu et al. compararon tres preprocesamientos (histograma temporal, espectro de potencia y tiempo-frecuencia) y el espectrograma STFT fue claramente el mejor para redes convolucionales.

**Origen de los datos.** Varios estudios entrenan con simulaciones (HIDE & SEEK en Akeret et al.; MeerKAT en Vafaei Sadr et al.; señales B3I con SNR de −26 dB en Liu et al.; HERA en Pritchard et al.) porque ofrecen etiquetas perfectas y grandes volúmenes. Otros usan datos reales: mediciones FM húngaras públicas en Zenodo (Takács et al.), I/Q reales de espectro compartido (Tschimben y Gifford), dos conjuntos reales de *spoofing* y *jamming* (Ghanbarzade y Soleimani) y datos del telescopio KAT-7 etiquetados a mano (Vafaei Sadr et al.). La brecha simulación–realidad es uno de los problemas centrales, y el aprendizaje por transferencia de Vafaei Sadr et al. es la mejor estrategia documentada para salvarla.

**Etiquetado.** Coexisten tres estrategias: etiquetas perfectas heredadas de la simulación, etiquetado humano experto (caro y lento, como en KAT-7) y etiquetado automático o débil mediante *flags* estadísticos o indicadores de calidad del propio instrumento (Alam, 2024; Sharma et al., 2023).

**Datasets públicos mencionados en los artículos revisados:** el Hungarian FM Spectrum Measurements Dataset (Zenodo), usado por Takács et al., y los bancos de pruebas simulados de la comunidad de radioastronomía, como los derivados de HERA.

## 7. Análisis comparativo de los artículos revisados

**Akeret et al. (2017)**, *Radio frequency interference mitigation using deep convolutional neural networks* (Astronomy and Computing, 18). Aborda el marcado de RFI en datos tiempo-frecuencia con una U-Net entrenada con simulaciones y datos reales de Bleien. Su evaluación compara la exactitud contra SumThreshold y muestra precisión competitiva. *Limitaciones:* validado en un solo observatorio y con volúmenes moderados; arquitecturas posteriores lo superaron. *Aporte:* demuestra que un espectrograma puede tratarse como imagen y deja código público documentado, útil como modelo de referencia.

**Vafaei Sadr et al. (2020)**, *Deep learning improves identification of Radio Frequency Interference* (MNRAS, 499). Propone R-Net, una CNN residual entrenada con simulaciones de MeerKAT y afinada con datos reales de KAT-7 etiquetados a mano. Con AUC, F1 y MCC supera al *flagger* de MeerKAT; con transferencia alcanza AUC 0,91 frente a 0,67 sin ella. *Limitaciones:* la fase sin calibrar aporta casi nada de información respecto a la amplitud, y la afinación requiere datos reales etiquetados. *Aporte:* la estrategia entrenar-simulado/afinar-real.

**Czech e Inggs (2018)**, *A CNN and LSTM-based approach to classifying transient radio frequency interference* (Astronomy and Computing, 25). Clasifica RFI transitorios combinando CNN y LSTM. *Aporte:* establece que las interferencias transitorias requieren modelos con memoria temporal. *Nota:* este artículo se revisó solo a partir de su resumen; los detalles de datos y métricas no se reportan aquí.

**Tschimben y Gifford (2022)**, *Anomaly Detection with Autoencoders for Spectrum Sharing and Monitoring*. Detecta transmisiones no autorizadas con un autoencoder profundo y un LSTM-autoencoder no supervisados sobre grabaciones I/Q reales. Evalúa la identificación de anomalías en tiempo y frecuencia. *Limitaciones:* los resultados dependen de qué se considere "normal" en el entrenamiento y se aplicó a un entorno concreto. *Aporte:* prueba que se puede monitorizar el espectro sin etiquetas.

**Takács et al. (2026)**, *Autoencoder-Based FM Spectrum Anomaly Detection: A Pipeline-Level Study of Preprocessing, Model Capacity, and Reconstruction Scoring* (Sensors, 26(18)). Detecta anomalías de radiodifusión FM sobre el Hungarian FM Spectrum Measurements Dataset (cascadas de agua de la banda VHF II estandarizadas a 5 kHz y 400 s). Compara cinco variantes de autoencoder y un VAE convolucional, evaluadas con F1 y recall por el fuerte desbalance de clases. La mejor configuración alcanzó F1 de 91,4 % en desarrollo, pero en la partición de prueba el rendimiento cayó y cambió el ranking, señal de riesgo de sobreajuste. *Limitaciones:* desbalance extremo, escasez y poca diversidad de anomalías, necesidad de expertos para anotar. *Aporte:* dataset público real y un pipeline completo, replicable parcialmente.

**Ghanbarzade y Soleimani (2025)**, *GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning* (arXiv:2501.02352). Detecta e identifica *jamming* y *spoofing* sobre dos conjuntos reales, combinando Random Forest, árboles de decisión y regresión logística con aprendizaje profundo y visión por computador. Reporta ~99 % de exactitud en *jamming*, unos cinco puntos sobre trabajos previos. *Limitaciones:* el rendimiento en *spoofing* parece inferior por la dificultad de distinguir señales falsas muy similares a las reales (el resumen no detalla todas las limitaciones). *Aporte:* los algoritmos clásicos siguen siendo competitivos con buenas características.

**Liu et al. (2025)**, *Deep learning-based GNSS composite jamming detection and recognition technology* (Frontiers in Signal Processing, 3:1567926). Aborda interferencias compuestas con señales B3I simuladas (SNR −26 dB) que combinan cinco tipos. Compara ResNet-18 con LcxNet-Fusion; el preprocesamiento STFT fue el mejor, con sensibilidad de −20 dB JNR al 90 % de detección, y la red propia redujo parámetros un 43 % y tiempo un 15 %. *Limitaciones:* los datasets públicos GNSS solo cubren cinco tipos, el solapamiento de tipos compuestos sigue siendo difícil y las CNN heredadas de ImageNet no están optimizadas para esta tarea. *Aporte:* cuantifica el efecto del preprocesamiento y justifica las redes ligeras para tiempo real.

**Alam (2024)**, tesis *Deep learning-based radio frequency interference detection and mitigation for microwave radiometers with 2-D spectral features* (Mississippi State University). Aplica CNN a datos crudos de radiómetros, etiquetados con *quality flags*, para detectar RFI de radar, 5G y drones a escala global. Mejora la precisión y la generalización frente a detectores artesanales por tipo. *Limitaciones:* depende de la calidad de los *quality flags* usados como etiquetas. *Aporte:* ejemplo de etiquetado débil automático.

## 8. ¿Cómo se evalúan los resultados?

Las métricas varían con la formulación del problema:

- **Clasificación binaria o multiclase** (¿hay interferencia? ¿de qué tipo?): exactitud, precisión, recall y F1, con la matriz de confusión como resumen. Takács et al. (2026) justifican usar F1 y recall en lugar de exactitud porque el desbalance normal/anómalo hace que la exactitud sola sea engañosa.
- **Segmentación por píxel:** exactitud y F1 por píxel, AUC-ROC y coeficiente MCC (Vafaei Sadr et al., 2020; Pritchard et al., 2025), comparando contra *flaggers* clásicos como SumThreshold o el de MeerKAT.
- **GNSS:** tasas de detección e identificación en función de la relación jamming/ruido (JNR), es decir, la potencia mínima de interferencia que el modelo puede detectar (Liu et al., 2025).
- **Tiempo real:** número de parámetros y tiempo de inferencia (Liu et al., 2025; Pritchard et al., 2025).

No todos los estudios comparan contra la misma línea base ni reportan las mismas métricas, por lo que las cifras citadas **no son directamente comparables entre sí**: cada una debe leerse dentro de su propio experimento.

## 9. Limitaciones de los métodos existentes

**Datos.**
- *Escasez de datos etiquetados:* el etiquetado experto es caro (KAT-7 fue etiquetado a mano) y las anomalías genuinas son raras y poco diversas (Takács et al., 2026).
- *Brecha simulación–realidad:* entrenar solo con simulaciones degrada el rendimiento en datos reales; sin transferencia, el mejor modelo de Vafaei Sadr et al. (2020) cayó de AUC 0,91 a 0,67.
- *Desbalance de clases:* los eventos anómalos son una fracción mínima y ocupan regiones muy localizadas del espectrograma, lo que exige métricas apropiadas (Takács et al., 2026).

**Modelos.**
- *Generalización limitada:* un modelo entrenado en un telescopio, banda o receptor no traslada automáticamente su rendimiento a otro. Además, las CNN estándar, diseñadas para ImageNet, no están optimizadas para señales radioeléctricas (Liu et al., 2025).
- *Coste computacional:* las CNN profundas pueden impedir la operación continua en tiempo real, lo que motiva redes ligeras y espiking (Liu et al., 2025; Pritchard et al., 2025).
- *Interferencias compuestas:* siguen siendo el escenario más débil (Liu et al., 2025).
- *Entradas útiles restringidas:* la fase sin calibrar aporta casi nada frente a la amplitud (Vafaei Sadr et al., 2020).
- *Herencia de errores:* cuando las etiquetas provienen de *flaggers* clásicos o de *quality flags*, el modelo hereda los errores del método que lo entrenó (Alam, 2024).

**Evaluación.** La falta de estandarización de datasets, particiones y métricas impide comparaciones justas entre algoritmos.

Ninguna de estas limitaciones invalida la línea de trabajo, pero fijan expectativas realistas sobre qué se puede replicar con recursos académicos.

## 10. Respuestas a las preguntas de la investigación

**1. ¿Qué tipos de interferencias se pueden detectar mediante Machine Learning?**
Prácticamente todos los tipos estudiados en la literatura revisada: interferencias de banda estrecha (CW), barrido (chirp), pulsadas, de espectro adaptado y sus combinaciones (Liu et al., 2025); *jamming* y *spoofing* en GNSS (Ghanbarzade y Soleimani, 2025); transmisiones no autorizadas en espectro compartido (Tschimben y Gifford, 2022); anomalías de emisores FM (Takács et al., 2026); y RFI de radar, 5G y drones sobre radiómetros (Alam, 2024). La detección se ha demostrado tanto para interferencias persistentes como transitorias.

**2. ¿Qué algoritmos se utilizan y cómo se aplican?**
Tres familias. Los *clásicos supervisados* (Random Forest, árboles de decisión, regresión logística) trabajan sobre características numéricas. Los *métodos de detección de anomalías*, sobre todo autoencoders y LSTM-autoencoders, se entrenan solo con datos normales y delatan la interferencia mediante el error de reconstrucción. Las *redes profundas*, dominadas por CNN sobre espectrogramas, incluyen U-Net (segmentación), R-Net/ResNet (clasificación y segmentación con transferencia), CNN+LSTM (transitorios), redes ligeras y redes espiking. La aplicación típica consiste en convertir la señal en una imagen tiempo-frecuencia, entrenar con ejemplos etiquetados (o solo normales, en el caso no supervisado) y decidir píxel a píxel o ventana a ventana.

**3. ¿Qué datos necesitan los modelos?**
Los clásicos necesitan vectores de características (potencias, estadísticos). Los profundos funcionan mejor con espectrogramas o cascadas de agua obtenidos por STFT a partir de I/Q o de mediciones de potencia. Los métodos de anomalía requieren abundantes mediciones normales. Se necesitan etiquetas (reales, simuladas o automáticas mediante *flags*) salvo en el caso no supervisado, y los mejores resultados combinan datos simulados para el volumen y reales para la afinación.

**4. ¿Cómo evalúan los investigadores sus resultados?**
Con exactitud, precisión, recall y F1 en clasificación; F1 por píxel, AUC-ROC y MCC en segmentación; tasas de detección frente a JNR en GNSS; y parámetros y tiempo de inferencia cuando importa la eficiencia. Cada estudio usa su propia línea base y particiones, por lo que las métricas no son comparables directamente.

**5. ¿Qué limitaciones presentan los métodos existentes?**
Las descritas en la sección 9: escasez de datos etiquetados, brecha simulación–realidad, desbalance de clases, generalización limitada, coste computacional de las CNN profundas, dificultad con interferencias compuestas y falta de estandarización de las evaluaciones.

## 12. Referencias

1. Akeret, J.; Chang, C.; Lucchi, A.; Refregier, A. (2017). *Radio frequency interference mitigation using deep convolutional neural networks*. Astronomy and Computing, 18, 35–44. arXiv:1609.09077. https://arxiv.org/abs/1609.09077
2. Alam, A. M. (2024). *Deep learning-based radio frequency interference detection and mitigation for microwave radiometers with 2-D spectral features*. Tesis, Mississippi State University. https://scholarsjunction.msstate.edu/td/6215
3. Ayazbek, et al. *CNN compacta para detección y clasificación simultánea de interferencias.* [Referencia por completar: título, año y fuente.]
4. Czech, D.; Mishra, A.; Inggs, M. (2018). *A CNN and LSTM-based approach to classifying transient radio frequency interference*. Astronomy and Computing, 25, 52–57. https://www.sciencedirect.com/science/article/abs/pii/S2213133716301056
5. Ghanbarzade, A.; Soleimani, H. (2025). *GNSS/GPS Spoofing and Jamming Identification Using Machine Learning and Deep Learning*. arXiv:2501.02352. https://arxiv.org/abs/2501.02352
6. Liu, C.; et al. (2025). *Deep learning-based GNSS composite jamming detection and recognition technology*. Frontiers in Signal Processing, 3:1567926. https://doi.org/10.3389/frsip.2025.1567926
7. Offringa, A. R.; de Bruyn, A. G.; Biehl, M.; Zaroubi, S.; Bernardi, G.; Pandey, V. N. (2010). *Post-correlation radio frequency interference classification methods*. MNRAS, 405, 155.
8. Pritchard, N. J.; Wicenec, A.; Bennamoun, M.; Dodson, R. (2025). *Advancing RFI-Detection in Radio Astronomy with Liquid State Machines*. arXiv:2504.09796. https://arxiv.org/abs/2504.09796
9. Sharma, A. S.; et al. (2023). *Identification of RF Interference in Astronomical Observations* (supervisión débil). [Completar revista/fuente.]
10. Takács, S. L.; et al. (2026). *Autoencoder-Based FM Spectrum Anomaly Detection: A Pipeline-Level Study of Preprocessing, Model Capacity, and Reconstruction Scoring*. Sensors, 26(18), 5979. https://www.mdpi.com/1424-8220/26/18/5979
11. Tschimben, S.; Gifford, K. (2022). *Anomaly Detection with Autoencoders for Spectrum Sharing and Monitoring*. Proc. IEEE. https://par.nsf.gov/servlets/purl/10519014
12. Vafaei Sadr, A.; et al. (2020). *Deep learning improves identification of Radio Frequency Interference*. MNRAS, 499, 379–391. arXiv:2005.08992. https://arxiv.org/abs/2005.08992

---

*Nota:* las cifras citadas provienen de los resúmenes o textos de los artículos listados. Czech e Inggs (2018), Sharma et al. (2023) y Ayazbek et al. se revisaron de forma parcial y conviene completar sus datos antes de la entrega final.
