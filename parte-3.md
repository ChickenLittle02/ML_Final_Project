# Estado del arte: identificación de espacios en blanco del espectro con Machine Learning

## 1. ¿Qué son los espacios en blanco del espectro?

El espectro radioeléctrico se reparte entre servicios con licencia, como la televisión, la telefonía móvil, la radio o los radares. El problema es que esa asignación es fija, y en la práctica muchas bandas pasan buena parte del tiempo sin usarse en ciertos lugares o momentos. A esos huecos se les llama espacios en blanco, *spectrum white spaces* o *spectrum holes*. Son porciones del espectro que su usuario con licencia (el usuario primario) no está usando en ese instante y en esa zona, así que un usuario secundario podría aprovecharlas sin generar interferencia dañina. Esta es la idea base de la radio cognitiva y del acceso dinámico al espectro.

Lo importante es que un espacio en blanco no es una propiedad fija de una frecuencia. Depende del tiempo, de la ubicación y de la frecuencia al mismo tiempo, y cambia constantemente, y por eso es un problema difícil. Según Aygül et al. (2025), los sistemas de radio cognitiva necesitan sensar y predecir el espectro con buena precisión para poder encontrar y aprovechar estos huecos de forma eficiente.

## 2. ¿Cómo se identifica un espacio en blanco?

Identificar un espacio en blanco consiste en decidir si en una banda y un momento concretos hay solo ruido (banda libre) o hay también una señal del usuario primario (banda ocupada). Para esto se parte de muestras de la señal captadas por un receptor, por ejemplo con una radio definida por software (SDR) o un analizador de espectro. De esas muestras se calcula algún valor, como la energía, y se compara con un umbral. Elegir bien ese umbral es clave porque hay que equilibrar dos errores: la probabilidad de detección, que es acertar cuando la banda sí está ocupada, y la probabilidad de falsa alarma, que es decir que está ocupada cuando en realidad está libre.

Los métodos clásicos más conocidos son estos:

- **Detección de energía:** mide la potencia recibida y la compara con un umbral. Es sencilla y no necesita conocer la señal, pero funciona mal cuando la relación señal-ruido es baja y depende de conocer bien el nivel de ruido.
- **Filtro adaptado:** es el mejor método si se conoce la forma de onda del usuario primario, pero justamente requiere ese conocimiento previo.
- **Detección cicloestacionaria:** aprovecha que las señales moduladas tienen patrones periódicos que el ruido no tiene. Es más robusta, aunque más costosa de calcular.
- **Detección por valores propios:** analiza los autovalores de la matriz de covarianza de las muestras y no necesita conocer ni la señal ni el ruido.

A esto se suma el **sensado cooperativo**, donde varios nodos comparten lo que miden y combinan sus resultados. Así se reducen problemas como el desvanecimiento de la señal o el del terminal oculto. También existen las **bases de datos geolocalizadas**, como las usadas en los TV White Spaces, aunque estas informan lo que está registrado y no la ocupación real en cada momento.

## 3. ¿Qué técnicas de Machine Learning se han utilizado?

En la literatura el ML se usa para dos cosas. La primera es la detección, es decir, clasificar una banda como libre u ocupada. En lugar de fijar un umbral a mano, el modelo aprende a separar las dos clases a partir de características de la señal o directamente de los datos crudos. La segunda es la predicción, que busca estimar la ocupación futura usando el historial, y se plantea como un problema de series temporales.

**Árboles de decisión y Random Forest.** Se usan como clasificadores sobre vectores de características (energía, estadísticos, medidas de varios sensores, etc.). Los árboles son fáciles de interpretar y muy rápidos al momento de usarlos. Random Forest combina muchos árboles entrenados con distintas submuestras, lo que reduce el sobreajuste y suele dar buenos resultados con datos tabulares. Por eso aparecen mucho como modelo de comparación en los estudios. Su limitación es que no entienden el tiempo por sí solos, así que hay que darles variables rezagadas o ventanas deslizantes para que tengan en cuenta el pasado.

**SVM (máquinas de vectores de soporte).** Buscan el hiperplano que mejor separa las clases ocupado y libre, y con kernels no lineales, como el RBF, pueden manejar fronteras más complejas. Se han usado tanto en sensado individual como cooperativo, donde las medidas de varios usuarios forman el vector de entrada. Funcionan bien con conjuntos de datos pequeños o medianos, pero escalan peor cuando los datos son muchos y dependen bastante de la elección de los hiperparámetros.

**Redes neuronales.** Van desde el perceptrón multicapa hasta arquitecturas profundas. Las redes convolucionales (CNN) tratan el espectro como una imagen, con tiempo en un eje y frecuencia en el otro, y sacan patrones espaciales. Las redes recurrentes se encargan de la parte temporal. Aygül et al. (2025) concluyen que, desde redes neuronales simples hasta aprendizaje profundo avanzado, estos modelos mejoran la precisión porque aprovechan las correlaciones en tiempo y espacio de los datos del espectro. También se usan autoencoders, redes GAN para generar datos sintéticos y aprendizaje por refuerzo para decidir qué canal sensar o usar.

**Modelos de series temporales (LSTM).** Antes se usaban modelos estadísticos como ARIMA o los modelos ocultos de Markov, pero las LSTM manejan mejor las dependencias a largo plazo. Según una revisión de algoritmos de ML en redes de radio cognitiva, las LSTM son muy populares porque pueden recordar entradas y resultados anteriores. Esa misma revisión menciona trabajos que aprovechan la correlación espacio-temporal del historial de disponibilidad, otros que definen la longitud de la ventana de entrada analizando la autocorrelación de la ocupación, y modelos ConvLSTM de secuencia a secuencia para predicción a largo plazo. También se reporta que las redes con retardos temporales (TDNN) superaron a los modelos ARIMA estacionales en horizontes cortos. Hoy es común combinar ambos tipos: la CNN saca los patrones espaciales y la LSTM los temporales.

## 4. ¿Qué diferencia hay entre detectar una frecuencia libre y predecir su disponibilidad futura?

Detectar es un problema del presente. Con lo que se observa ahora se decide si la banda está libre u ocupada en este instante, y se mide con la probabilidad de detección y de falsa alarma. Predecir, en cambio, es un problema del futuro. Con el historial de ocupación se estima cómo estará la banda más adelante, sin necesidad de volver a medirla en ese momento.

Esta diferencia tiene consecuencias prácticas. Sensar gasta energía y tiempo, y mientras el nodo sensa no puede transmitir. Si se puede predecir, se reduce la cantidad de sensado, se elige de antemano a qué canal cambiar, se anticipa cuándo va a volver el usuario primario y se evitan interferencias y retardos. El costo es que la predicción añade un error propio, porque se basa en patrones del tráfico (por ejemplo, horarios del día o de la semana) que pueden cambiar sin aviso.

También cambian los datos y la forma de evaluar. La detección trabaja con muestras de señal de una ventana corta y es básicamente una clasificación. La predicción trabaja con secuencias largas de ocupación y es un pronóstico, donde importan el horizonte de tiempo y el hecho de que el error se acumula. En la práctica ambos se combinan, porque la detección alimenta el historial y la predicción reduce la necesidad de detectar todo el tiempo. Los métodos estadísticos tradicionales se quedan cortos porque la ocupación del espectro no es estacionaria, y esto se agrava con la movilidad y diversidad de usuarios en redes 6G.

## 5. ¿Qué datos se utilizan para entrenar los modelos?

Las fuentes de datos son principalmente tres. Primero están las **campañas de medición reales**, donde un analizador de espectro o una SDR barre un rango de frecuencias durante días o meses y guarda la potencia por canal y por instante, formando matrices tiempo-frecuencia. Las bandas más estudiadas son GSM, TV, LTE, la banda ISM de 2,4 GHz y la radio móvil terrestre. Segundo, están los **datos sintéticos o simulados**, generados con modelos de tráfico como cadenas de Markov ON/OFF o procesos de Poisson, o con simuladores de señales con SNR controlada. Tercero, los **conjuntos de datos abiertos**, como la familia RadioML, que se usan sobre todo para clasificar señales y modulaciones. La revisión de Aygül et al. (2025) incluso dedica una parte a las técnicas para generar datasets para predicción de ocupación.

Antes de entrenar, los datos normalmente se preprocesan: se binariza la potencia con un umbral (libre u ocupado) o se normaliza, se arman ventanas deslizantes y se separan los datos de entrenamiento y prueba de forma cronológica, para que el modelo no vea el futuro durante el entrenamiento. Además de la potencia, a veces se agregan la hora, el día de la semana, la ubicación o la ocupación de canales vecinos.

## 6. Aplicaciones de la identificación de espacios en blanco

La aplicación principal es la radio cognitiva y el acceso dinámico al espectro, donde usuarios secundarios usan los huecos sin molestar a los licenciados. Algunos ejemplos son:

- Los TV White Spaces, para llevar internet de banda ancha a zonas rurales.
- La coexistencia entre Wi-Fi, LTE/NR sin licencia y otras tecnologías en bandas compartidas.
- Las redes IoT y de sensores, que ahorran energía y evitan interferencia eligiendo canales libres.
- Las redes 5G y 6G, para compartir espectro y gestionar recursos.
- Las comunicaciones militares y de emergencia, que necesitan cambiar rápido a canales disponibles.
- Las redes vehiculares, satelitales y de drones.
- El monitoreo y la regulación del espectro, porque los mapas de ocupación ayudan a ver qué bandas están subutilizadas o a detectar transmisores no autorizados.

También hay una aplicación en seguridad: la revisión de 2025 trata amenazas que afectan a la radio cognitiva y los métodos de ML usados para detectarlas.

## 7. ¿Qué errores cometen los modelos y qué limitaciones presentan?

**Errores de decisión.** Los dos errores principales son la falsa alarma, que hace perder oportunidades de usar un canal libre, y la detección fallida, que consiste en decir que la banda está libre cuando no lo está. Este segundo error es el más grave porque causa interferencia al usuario primario, y las regulaciones suelen ser muy estrictas con eso. Por eso los modelos se suelen ajustar para minimizarlo, aunque se pierda algo de eficiencia.

**Dependencia de la SNR.** Cuando la relación señal-ruido es muy baja, tanto los métodos clásicos como los de ML empeoran. Además, un modelo entrenado en un cierto rango de SNR puede fallar fuera de ese rango.

**Generalización y cambios en los datos.** Un modelo entrenado con datos de una banda o una ciudad puede rendir peor en otra. La ocupación cambia con el tiempo y aparecen eventos que no estaban en el historial (festivos, emergencias, nuevos operadores, cambios regulatorios). Esto se conoce como *concept drift* y obliga a reentrenar o adaptar el modelo continuamente.

**Errores propios de la predicción.** El error crece cuanto más lejos se predice, porque cada predicción se apoya en las anteriores. Además, los datos suelen estar desbalanceados, ya que muchas bandas están casi siempre libres o casi siempre ocupadas, así que una exactitud alta puede esconder un mal desempeño en la clase poco frecuente. Por eso conviene usar métricas como precisión, exhaustividad (*recall*), F1 o AUC. Y si el tráfico es muy aleatorio, simplemente es difícil de predecir.

**Costo computacional.** Las redes profundas necesitan muchos recursos para entrenar y para funcionar, lo cual es un problema en dispositivos pequeños con poca batería, como los sensores IoT. Random Forest y SVM son más ligeros, pero menos capaces de modelar relaciones complejas en tiempo y espacio.

**Calidad y disponibilidad de datos.** Los datos reales etiquetados son escasos, hacer campañas de medición es caro y los datos sintéticos pueden no reflejar bien la realidad. Además, los sensores tienen ruido y problemas de calibración que contaminan las etiquetas.

**Interpretabilidad y seguridad.** Los modelos profundos son poco interpretables, y esto es un problema cuando se necesitan garantías de no interferencia. También son vulnerables a ataques, como la emulación del usuario primario (PUE), donde un atacante imita la señal primaria para que los demás dejen libre el canal, o la falsificación de datos en el sensado cooperativo.

**Evaluación poco comparable.** Cada estudio usa datos, bandas, horizontes y métricas distintos, por lo que es difícil comparar resultados entre trabajos. Entre las líneas futuras que menciona la revisión de 2025 está aprovechar mejor las correlaciones multidimensionales (tiempo, frecuencia y espacio) con aprendizaje profundo para redes 6G y posteriores.

## Conclusión

En resumen, el área pasó de los detectores con umbral y los modelos estadísticos a clasificadores de ML como árboles, Random Forest y SVM, y más recientemente a redes profundas (CNN, LSTM, ConvLSTM e híbridos) que aprovechan las correlaciones en tiempo, frecuencia y espacio. La diferencia clave es que detectar dice si una banda está libre ahora, mientras que predecir intenta anticipar si lo estará más adelante, lo que es más útil en la práctica pero tiene más incertidumbre. Los retos que siguen abiertos son la generalización a otros entornos, la falta de datos reales representativos, el costo computacional, la seguridad frente a ataques y la necesidad de garantizar interferencia mínima al usuario primario.
