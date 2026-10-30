# Estado del Arte: Monitoreo y Gestión del Espectro Radioeléctrico mediante Aprendizaje Automático

**Proyecto:** Detección de Interferencias e Identificación de Espacios en Blanco

---

# Sección 1: Fundamentos del monitoreo y la gestión del espectro radioeléctrico

## 1.1. Introducción y objetivo

Casi todas las tecnologías inalámbricas que usamos a diario (como la radio, la televisión, los teléfonos celulares, el Wi-Fi, el Bluetooth, los sistemas de posicionamiento por satélite, los radares y las redes de sensores) transmiten sus señales a través del mismo medio físico: el espectro radioeléctrico. Como cada día hay más aparatos conectados y más servicios, este recurso se tiene que organizar con cuidado para que todos puedan funcionar al mismo tiempo sin chocarse ni arruinarse la señal entre sí. De poner ese orden se encargan la gestión y el monitoreo del espectro.

En esta primera parte se buscan responder cuatro preguntas principales:

1. ¿Qué problemas intenta resolver el monitoreo del espectro?
2. ¿Por qué es importante detectar las interferencias e identificar los espacios en blanco?
3. ¿Qué aporta el aprendizaje automático (que en inglés llaman *Machine Learning*) a este campo?
4. ¿Qué relación hay entre detectar una interferencia y encontrar un espacio en blanco?

Para responder a esto, primero dejamos claros los conceptos básicos como el espectro, la gestión, el monitoreo, la ocupación, las interferencias, los espacios en blanco, la radio cognitiva y el acceso dinámico al espectro. Después explicamos por qué las formas tradicionales de revisar el espectro ya no alcanzan y por qué necesitamos usar modelos de aprendizaje automático.

## 1.2. ¿Qué es el espectro radioeléctrico?

El Reglamento de Radiocomunicaciones de la Unión Internacional de Telecomunicaciones define las ondas radioeléctricas como ondas electromagnéticas que viajan por el espacio sin necesidad de un cable o una guía artificial. El espectro radioeléctrico es el conjunto de todas esas frecuencias que sirven para enviar y recibir información a distancia.

El espectro se considera un recurso natural limitado. No se gasta por el hecho de usarlo, pero si dos antenas transmiten en la misma frecuencia, en el mismo lugar y al mismo tiempo, las señales se van a amontonar y se van a arruinar mutuamente. Por eso siempre se analiza pensando en tres cosas al mismo tiempo: la frecuencia, el tiempo y el espacio.

Las propiedades físicas de cómo viaja la señal cambian mucho según la frecuencia:

En la banda de Muy Alta Frecuencia (que va de 30 a 300 Megahercios), la señal viaja distancias largas y atraviesa bien los obstáculos. Se usa en la radio comercial en Frecuencia Modulada, las radios de los aviones y las comunicaciones de emergencia.

En la banda de Ultra Alta Frecuencia (que va de 300 Megahercios a 3 Gigahercios), se ubican la televisión digital, la telefonía celular, las señales de posicionamiento por satélite, el Wi-Fi común y el Bluetooth. Aquí es donde más problemas de amontonamiento de señales ocurren porque casi todos los aparatos cotidianos transmiten en este rango.

En la banda de Súper Alta Frecuencia (que va de 3 a 30 Gigahercios), están las redes de quinta generación, el Wi-Fi más rápido de 5 Gigahercios, los radares y las antenas parabólicas satelitales. Estas frecuencias soportan enviar una cantidad enorme de datos por segundo, pero la señal no viaja muy lejos y cualquier pared o lluvia fuerte la apaga.

## 1.3. ¿Qué significa gestionar y monitorear el espectro?

### 1.3.1. Gestión del espectro

La gestión del espectro reúne los trámites administrativos y las reglas técnicas con las que las autoridades deciden quién puede usar cada frecuencia, en qué ciudad o zona, a qué horas y con cuánta potencia. Esto se organiza en tres niveles principales:

El primer nivel es la atribución, donde los organismos internacionales reservan una banda de frecuencias para un tipo de servicio (por ejemplo, reservan un bloque para telefonía móvil y otro para aviación).

El segundo nivel es la adjudicación, donde se distribuyen los canales entre los diferentes países o regiones.

El tercer nivel es la asignación, donde la autoridad de telecomunicaciones de cada país le da el permiso oficial a una empresa o institución para que use una frecuencia específica con una antena en un lugar concreto.

El modelo tradicional para dar estos permisos se conoce como mando y control. Consiste en vender o entregar licencias exclusivas y fijas. Esto significa que si una empresa compra una frecuencia, solo ella puede transmitir ahí, incluso si casi no la usa durante todo el día. Aunque este método evita que la gente se choque en la frecuencia, termina desperdiciando muchísimo espacio libre.

### 1.3.2. Monitoreo del espectro

El monitoreo del espectro consiste en salir a medir y revisar con antenas y equipos de recepción qué es lo que de verdad está pasando en el aire. Sus metas principales son:

Comprobar que las empresas con licencia respeten la potencia y la frecuencia que les permitieron usar.

Detectar e investigar de dónde vienen las interferencias cuando los usuarios se quejan de que su servicio anda mal.

Medir cuántas horas al día se usa cada canal para saber qué frecuencias están saturadas y cuáles están completamente vacías.

Descubrir antenas piratas o transmisiones no autorizadas.

Hoy en día, para no gastar fortunas en equipos de laboratorio gigantes, este monitoreo se hace con aparatos llamados Radios Definidas por Software (que son receptores pequeños que se conectan a una computadora y hacen todo el procesamiento por programas) y con redes de sensores económicos distribuidos por toda una ciudad.

## 1.4. ¿Qué es la ocupación espectral?

La ocupación espectral es un porcentaje que nos dice cuánto tiempo está siendo usada una frecuencia. Para medirla, un receptor escucha esa frecuencia muchas veces seguidas. Si en una medición la potencia que capta supera un límite fijado (que llamamos umbral de energía), esa medición se marca como ocupada. Al final, se divide el número de veces que dio ocupado entre el número total de mediciones realizadas.

De esto surgen dos detalles muy importantes para nuestro proyecto:

En primer lugar, la ocupación depende totalmente de cómo está configurado el receptor. Si ponemos el umbral muy alto, las señales débiles van a parecer canal libre cuando en verdad hay alguien hablando despacio. Si lo ponemos muy bajo, el ruido de fondo de la electricidad de la ciudad va a hacer parecer que el canal está ocupado cuando en verdad está vacío.

En segundo lugar, la ocupación es una etiqueta práctica, no una verdad perfecta. Si un conjunto de datos viene etiquetado como ocupado o libre usando un umbral simple, los modelos de aprendizaje automático van a aprender ese criterio con los mismos errores que traía la medición original.

### La paradoja de la escasez

Cuando los científicos salen a medir las ciudades con receptores de radio, descubren algo insólito: el espectro parece que se está acabando porque casi todas las licencias ya tienen dueño, pero en la vida real las frecuencias están vacías casi todo el tiempo. Varios estudios en ciudades grandes demostraron que en promedio solo se usa entre el 13% y el 17% del espectro disponible por debajo de los 3 Gigahercios.

El problema real no es que no quepan más señales en la naturaleza, sino que el sistema de licencias es tan rígido que no le deja a nadie usar una frecuencia aunque el dueño no la esté usando en ese instante.

## 1.5. ¿Qué son las interferencias?

Una interferencia ocurre cuando una señal no deseada se mete en la frecuencia de la señal que a nosotros nos interesa escuchar, haciendo que la música o la llamada se escuche con lluvia, se entrecorte o se corte del todo. Cuando una interferencia es tan fuerte que tumba por completo un servicio importante (como la navegación de los aviones o las emergencias), se le llama interferencia perjudicial.

Las interferencias se pueden clasificar según varios criterios:

Por la intención con la que se generan, están las no intencionales (un aparato viejo roto que echa chispas electromagnéticas, una antena mal calibrada) y las intencionales (bloqueadores de señal o *jammers* que la gente usa ilegalmente para tapar los sistemas de rastreo por satélite de los carros o los celulares).

Por su forma en la frecuencia, existen las interferencias de canal adyacente (cuando una emisora muy potente invade el canal de al lado por no tener buenos filtros) y las interferencias de co-canal (cuando dos antenas usan exactamente la misma frecuencia al tiempo en zonas que se cruzan).

Por su aspecto visual al verlas en una gráfica, pueden ser tonos puros (una línea recta continua), ráfagas cortas (pulsos de señal), barridos de frecuencia (líneas diagonales llamadas *chirps* que cambian de frecuencia rápidamente) o ruido blanco de banda ancha (una mancha que ensucia todo el espectro).

## 1.6. ¿Qué son los espacios en blanco?

Un espacio en blanco (o hueco del espectro) es un canal que le pertenece legalmente a una empresa o servicio, pero que en un momento y en una ciudad determinada no está siendo utilizado por ellos.

El investigador Simon Haykin propuso una forma muy clara de clasificar las frecuencias según lo que captamos en el aire:

Los espacios negros son aquellos donde hay transmisiones locales muy potentes funcionando activamente.

Los espacios grises son los que tienen señales débiles o lejanas que se escuchan a medias.

Los espacios blancos son canales donde no se escucha absolutamente ninguna transmisión de radio, quedando libre únicamente el ruido de fondo natural.

Aprovechar estos espacios blancos fue lo que dio origen a proyectos regulados como los Espacios en Blanco de la Televisión, donde los canales de televisión vieja que quedaron abandonados se reutilizan para llevar internet de banda ancha a escuelas rurales sin causarle interferencia a nadie.

## 1.7. Radio cognitiva y acceso dinámico al espectro

### 1.7.1. Radio cognitiva

Una radio cognitiva es un sistema de comunicaciones inteligente que es capaz de mirar su entorno de radiofrecuencia, aprender de lo que escucha a su alrededor mediante modelos de computadora, y cambiar de manera automática su frecuencia, su potencia o la forma en que codifica los datos para no molestar a nadie y lograr la mejor conexión posible.

En este tipo de redes existen dos tipos de usuarios:

El usuario primario es el dueño legal del canal, la persona o empresa que pagó por la licencia. Tiene prioridad absoluta y nunca debe sufrir interferencias.

El usuario secundario es un usuario oportunista que no tiene licencia comercial, pero aprovecha los ratos en que el usuario primario está en silencio para enviar sus datos rápidamente, prometiendo salirse inmediatamente en cuanto el dueño regrese.

### 1.7.2. Acceso dinámico al espectro

El acceso dinámico al espectro es la estrategia opuesta a dar licencias fijas para siempre. En lugar de eso, el sistema va reasignando el uso de las frecuencias segundo a segundo según la necesidad real. Hay dos formas de hacerlo: la transmisión subyacente (donde el usuario secundario habla muy despacito al mismo tiempo que el primario, procurando que su voz se confunda con el ruido) y la transmisión oportunista (donde el secundario habla a todo volumen pero únicamente cuando el primario está en silencio total).

## 1.8. Limitaciones de los métodos tradicionales de monitoreo

Durante décadas, para saber si una frecuencia estaba libre u ocupada se usaron reglas matemáticas simples basadas en la energía. Sin embargo, estas técnicas tienen fallas graves cuando las llevamos a la práctica:

La detección de energía por umbral falla cuando la señal viene muy débil. Si el ruido del entorno cambia un poco por el calor o por la interferencia eléctrica de la calle, el sistema se confunde por completo. Esto se conoce en la literatura como el muro de la Relación Señal a Ruido: por debajo de cierta potencia, este método es totalmente incapaz de saber si hay una señal o solo ruido.

Los métodos avanzados como el filtro adaptado o la detección cicloestacionaria exigen que el receptor conozca de antemano la forma exacta de la señal que está buscando. Esto significa que no sirven para descubrir señales desconocidas ni para encontrar interferencias piratas raras.

El monitoreo con un solo sensor sufre del problema del terminal oculto: un receptor situado detrás de un edificio grande puede creer que la frecuencia está totalmente libre porque la montaña le tapa la señal, y al ponerse a transmitir termina arruinando la recepción de las personas que estaban del otro lado de la montaña.

Por último, poner a un técnico especialista a revisar las gráficas a mano con un analizador de espectro ya no es viable. Redes modernas de monitoreo generan decenas de Gigabytes de datos en una sola semana, por lo que la inspección humana no puede escalar.

## 1.9. ¿Por qué se utiliza el aprendizaje automático?

El aprendizaje automático nos permite reemplazar las reglas fijas escritas por un programador por modelos matemáticos que aprenden a reconocer patrones directamente mirando los datos históricos de las mediciones de radio.

Esto aporta ventajas aplastantes:

En primer lugar, los modelos pueden tomar decisiones sin necesidad de que les expliquemos la física exacta del canal ni el nivel preciso de ruido que hay en la calle.

En segundo lugar, pueden combinar al mismo tiempo las mediciones de diez o veinte sensores repartidos por la ciudad, resolviendo el problema de la montaña o el terminal oculto de forma automática.

En tercer lugar, cuando convertimos las señales de radio en imágenes llamadas espectrogramas, las redes neuronales convolucionales (que son las mismas que se usan para reconocer rostros en los celulares) son capaces de detectar la huella digital de cada tipo de señal, incluso cuando viene tapada por un nivel alto de ruido.

En cuarto lugar, nos permiten usar técnicas no supervisadas llamadas Autoencoders. Estos modelos aprenden cómo se ve el espectro en un día totalmente limpio y normal; cuando aparece cualquier señal rara o interferencia nueva que jamás habían visto, la marcan de inmediato como una anomalía.

## 1.10. Relación entre la detección de interferencias y la identificación de espacios en blanco

Aunque en este trabajo estudiamos la detección de interferencias y la búsqueda de espacios en blanco como dos temas separados, en el fondo son exactamente las dos caras de la misma moneda. Ambas se hacen las mismas preguntas: ¿qué señal hay presente en esta frecuencia, en este instante y en esta esquina?

Se relacionan en cuatro puntos clave:

Ambas parten exactamente de la misma materia prima: las muestras de voltaje o de potencia captadas por la antena.

Un error al buscar un espacio en blanco causa directamente una interferencia. Si el modelo dice que el canal está desocupado cuando en verdad el dueño estaba hablando despacito, el usuario secundario se va a meter al canal y le va a pisar la señal al dueño.

Un canal que tiene una interferencia fuerte nunca puede considerarse un espacio en blanco útil, porque los datos que enviemos por ahí no van a llegar bien.

Ambas tareas necesitan primero construir un modelo que entienda cómo se comporta el tráfico normal de la ciudad para poder juzgar si lo que ocurre en el aire es un silencio seguro o una anomalía peligrosa.

---

# Sección 2: Estado del arte sobre la detección de interferencias

## 2.1. Tipos de interferencias en la literatura

En los estudios científicos recientes sobre el tema, se abordan distintos tipos de interferencias según la aplicación concreta:

En el área de radioastronomía, las antenas gigantes que escuchan el espacio profundo captan señales mil millones de veces más débiles que un teléfono celular. Por eso, cualquier señal de radio FM, Wi-Fi o avión arruina las fotos del universo. A este problema se le llama marcado de interferencias de radiofrecuencia.

En los sistemas de navegación por satélite (como el Sistema de Posicionamiento Global), los investigadores se enfocan en detectar el bloqueo intencional (donde un transmisor pirata emite ruido fuerte para tapar la señal de los satélites) y el engaño o *spoofing* (donde el atacante transmite señales falsas pero parecidas a las verdaderas para hacerle creer a un barco o dron que está en otro país).

En el monitoreo comercial del espectro y de la radio FM, las investigaciones se centran en detectar fallas técnicas de los equipos, como estaciones de radio que transmiten fuera de su frecuencia permitida, interrupciones bruscas de señal o pérdida de la modulación de audio.

## 2.2. Técnicas de aprendizaje automático para detectar interferencias

Las técnicas empleadas por la comunidad científica se dividen en tres grandes grupos:

El primer grupo son los algoritmos clásicos supervisados, como los Árboles de Decisión, el Bosque Aleatorio (*Random Forest*) y las Máquinas de Vectores de Soporte. Estos modelos trabajan sobre tablas de datos numéricos donde previamente un ingeniero calculó promedios, desviaciones y picos de potencia. Son modelos sumamente rápidos de ejecutar, requieren poca memoria y logran porcentajes de acierto superiores al 98% cuando las características están bien calculadas. Su desventaja es que si la interferencia cambia ligeramente de forma, hay que volver a diseñar los cálculos matemáticos a mano.

El segundo grupo es la detección de anomalías sin supervisar. Aquí la técnica estrella son los Autoencoders. Estas redes neuronales se entrenan dándoles únicamente miles de grabaciones de espectro totalmente limpio. La red aprende a comprimir esa información limpia en pocas variables y luego a volver a dibujarla. Cuando el modelo se encuentra con una señal contaminada con interferencia, al intentar redibujarla comete un error enorme. Ese error de reconstrucción se usa como una alarma para avisar que algo raro está pasando en el aire sin necesidad de haber conocido la interferencia de antemano.

El tercer grupo son las redes neuronales profundas para imágenes, dominadas por las Redes Neuronales Convolucionales (como la arquitectura U-Net o la red ResNet). Estas redes toman las señales de radio y las convierten en imágenes de color llamadas espectrogramas (donde el eje horizontal es el tiempo, el eje vertical es la frecuencia y el color es la potencia). La red procesa el espectrograma tal cual como si fuera una fotografía y encierra en un recuadro o pinta píxel por píxel exactos los pedazos de la imagen que corresponden a una interferencia. Estudios destacados demostraron que mediante el aprendizaje por transferencia (entrenar primero la red neuronal con millones de simulaciones por computadora y luego ajustar los últimos detalles con unas cuantas fotos de antenas reales) se logran niveles de detección casi perfectos.

## 2.3. Datos empleados y limitaciones de la detección de interferencias

Para alimentar estos modelos, los investigadores usan cuatro formas de representar las señales: las muestras de Fase y Cuadratura (que son las lecturas complejas de voltaje directo que saca la antena), los vectores de densidad espectral de potencia, los parámetros de Stokes (usados en astronomía) y los espectrogramas generados por la Transformada de Fourier de Tiempo Corto.

Las principales limitaciones que enfrenta la literatura actual son tres:

La brecha entre la simulación y la realidad: la mayoría de los investigadores entrenan sus modelos con datos simulados por computadora porque son gratis y fáciles de etiquetar. Sin embargo, cuando esos mismos modelos se ponen a trabajar con antenas reales en la calle, su capacidad de detección cae estrepitosamente debido a las imperfecciones de los circuitos electrónicos y a los rebotes de las paredes.

El desbalance extremo de clases: en la vida real, las interferencias son eventos raros. El 99% del tiempo la señal transcurre de forma normal y solo un 1% del tiempo ocurre la falla. Si no se compensa esto al entrenar, los modelos tienden a volverse perezosos y a decir siempre que todo está normal para acertar casi siempre.

El costo computacional: las redes convolucionales profundas son gigantescas. Si queremos poner un detector de interferencias adentro de un dron o de un sensor pequeño alimentado por batería, muchas veces la computadora no tiene la potencia suficiente para procesar las fotos del espectro en tiempo real.

---

# Sección 3: Estado del arte sobre la identificación de espacios en blanco

## 3.1. Detección instantánea frente a predicción futura

A la hora de usar aprendizaje automático para buscar huecos libres en el espectro, los científicos separan claramente dos problemas distintos:

Por un lado está la detección instantánea, que es un problema del presente. El receptor escucha lo que pasa en este preciso segundo en una frecuencia y decide mediante una clasificación binaria entre dos opciones: el canal está Libre o el canal está Ocupado.

Por otro lado está la predicción temporal, que es un problema del futuro. El modelo no escucha el canal en este momento, sino que analiza la historia de las últimas cinco horas de transmisión y usa un modelo de series temporales para predecir si dentro de diez minutos el canal va a estar libre o no.

Predecir el futuro tiene una ventaja enorme: el sensor de radio no tiene que estar prendido gastando batería y midiendo a cada milisegundo, sino que se puede apagar para ahorrar energía y encender justo en el instante en que el modelo predijo que el dueño se iba a ir a almorzar.

## 3.2. Algoritmos utilizados para encontrar y predecir espacios en blanco

Para la detección del presente, los clasificadores más comunes siguen siendo las Máquinas de Vectores de Soporte y el Bosque Aleatorio. Estos modelos miran la potencia de varios sensores al mismo tiempo y trazan una frontera matemática para separar los momentos en que la potencia pertenece al ruido de cuando pertenece a una transmisión real.

Para la predicción del futuro, los métodos antiguos (como las Cadenas de Markov o los modelos autorregresivos estadísticos) han sido aplastados por las redes neuronales recurrentes, en especial la Memoria a Corto y Largo Plazo (conocida internacionalmente como red LSTM). Estas redes cuentan con bucles de memoria interna que les permiten recordar patrones del comportamiento humano, como por ejemplo que una banda de radio móvil se llena a las 8 de la mañana cuando la gente entra a trabajar y se desocupa al mediodía, ajustando sus predicciones a esos ritmos temporales.

## 3.3. Errores y consecuencias al clasificar espacios en blanco

En la búsqueda de canales libres se cometen dos tipos de errores que no valen lo mismo:

El primer error es la falsa alarma. Ocurre cuando el modelo dice que la frecuencia está ocupada cuando en verdad estaba totalmente vacía. La consecuencia de esto es simplemente una pérdida de oportunidad: la radio cognitiva se queda callada sin necesidad y perdemos la oportunidad de enviar nuestros datos rápido.

El segundo error es la detección fallida. Ocurre cuando el modelo dice que la frecuencia está completamente libre cuando en realidad el usuario primario sí estaba hablando. Este es el error más grave de todos porque hace que la radio secundaria transmita encima de la señal original, generando una interferencia perjudicial y violando las leyes de telecomunicaciones. Por eso, en los modelos de aprendizaje automático orientados a espacios en blanco, la sensibilidad de decisión siempre se ajusta para apretar al máximo la detección de señales reales, aceptando sufrir un poco más de falsas alarmas con tal de jamás pisarle la señal al dueño legal.

---

# Sección 4: Datos y conjuntos de datos para el entrenamiento

## 4.1. Tipos de datos disponibles y qué significan

Para poder entrenar cualquier modelo sin necesidad de salir a la calle a gastar fortunas comprando antenas y receptores caros, la comunidad científica internacional ha publicado varios conjuntos de datos abiertos.

Es importante entender qué contiene cada tipo de archivo:

Los archivos de densidad espectral de potencia guardan únicamente números que representan cuántos milivatios de energía captó la antena en cada pequeña franja de frecuencia a lo largo del día.

Los archivos de muestras de Fase y Cuadratura (conocidas como muestras I/Q) guardan los números complejos que representan la onda eléctrica cruda tal cual como pegó contra la antena. Tienen toda la información física de la señal, pero pesan muchísimos Gigabytes.

Los archivos de formato de metadatos de señal (SigMF) son un estándar internacional moderno que guarda las señales junto con un archivo de texto donde se explica detalladamente a qué hora se grabó, en qué ciudad, con qué antena, con qué ganancia y en qué frecuencia central.

## 4.2. Catálogo de conjuntos de datos públicos disponibles

A continuación se revisan los conjuntos de datos abiertos más relevantes que podemos usar directamente en una computadora sin tener antenas de radio físicas:

**LOG-a-TEC (`ws_traffic_20170606`):**
Contiene mediciones reales de densidad espectral de potencia grabadas durante 24 horas continuas en la banda europea de 868 Megahercios. Ofrece un ancho de banda de 192 Kilohercios dividido en 1024 celdas de frecuencia. Lo más valioso de este conjunto es que viene acompañado por archivos donde dos ingenieros expertos revisaron las gráficas y anotaron a mano dónde comienza y dónde termina cada transmisión real. Es un archivo sumamente liviano en formato de líneas de texto JSON, ideal para empezar a probar modelos de detección.

**Radio-Frequency Spectrum-Sharing Challenge (RFSSC):**
Es un conjunto de escenarios simulados creados para una competencia internacional. Proporciona escenas sintéticas con señales de radar y de radio compitiendo por el espectro. Lo relevante de este conjunto es que trae matrices binarias perfectas donde se indica con ceros y unos la verdad absoluta de qué canal está ocupado y si a un usuario secundario le estaría permitido transmitir o no bajo las reglas del juego.

**AERPAW Packapalooza 2025:**
Es una grabación real realizada en Estados Unidos usando antenas montadas en globos helostáticos y carros durante un festival urbano de seis horas. Mide frecuencias de barrido amplio hasta los 6 Gigahercios junto con las posiciones del sistema de posicionamiento satelital. El archivo comprimido pesa únicamente 438 Megabytes y se descarga directamente en un repositorio académico abierto. Es perfecto para construir mapas de ocupación reales, aunque requiere que nosotros mismos definamos dónde poner el umbral de energía para separar lo libre de lo ocupado.

**AERPAW Nodos Fijos:**
Contiene mediciones reales continuas captadas durante una semana completa por tres torres fijas de sensores que barrieron frecuencias desde los 87 Megahercios hasta los 6 Gigahercios. Los archivos suman 52 Gigabytes en total. Son excelentes para estudiar cómo cambia el tráfico de radio entre el día y la noche, pero requieren una computadora con mucho disco duro y memoria para poder abrirlos sin congelar el sistema.

**Dataset de Laboratorio para GNSS de Fraunhofer:**
Contiene capturas reales de muestras de Fase y Cuadratura grabadas en laboratorios especializados con señales de navegación por satélite. Su subconjunto principal (FIOT) incluye 9 clases etiquetadas con absoluta precisión: una clase de señal limpia sin interferencia y ocho clases con distintas interferencias (como ruido, barridos *chirp*, saltos de frecuencia, pulsos y tonos). El repositorio oficial declara unos 700 Megabytes de datos, pero hay que tener cuidado porque el archivo comprimido principal es muy pequeño (221 Kilobytes) ya que funciona mediante una herramienta de descarga llamada Git LFS que va bajando los archivos pesados uno a uno desde un servidor externo.

**Escenarios de Jamming en Interiores (Indoor Jamming Scenarios):**
Ofrece muestras reales en formato de MATLAB (.mat) grabadas dentro de oficinas con señales de modulación digital afectadas por interferencias de tono puro y ruido blanco a diferentes distancias y potencias. El conjunto completo pesa 118 Gigabytes distribuidos en 31 archivos individuales. Es una fuente hiperrealista pero exige descargar únicamente un par de archivos de ejemplo para no llenar el disco duro.

**Dataset Extendido de Espectrogramas de Fraunhofer:**
Contiene 30,000 imágenes sintéticas de espectrogramas que simulan el choque y la convivencia entre transmisiones de redes inalámbricas de área local (Wi-Fi), conexiones de corto alcance (Bluetooth) y redes móviles de quinta generación (5G). Pesa 19.6 Gigabytes y es ideal si se quiere entrenar redes de detección de objetos en imágenes.

**NIST CBRS Sharing Ecosystem Assessment:**
Es un repositorio gigante mantenido por el gobierno de Estados Unidos con mediciones reales de potencia en la banda compartida de 3.5 Gigahercios recopiladas entre los años 2024 y 2026. Muestra cómo conviven los radares de la marina con las antenas celulares comerciales en la vida real.

**RadioML 2016.10A:**
Es el dataset más famoso en internet para aprender a clasificar modulaciones de radio con muestras sintéticas. Sin embargo, se aclara fuertemente que **no debe ser el dataset principal de este proyecto** porque sus etiquetas indican únicamente qué tipo de modulación digital tiene la señal (como BPSK o QAM) y no si el canal está desocupado o si existe una interferencia en el aire.

---

# Sección 5: Algoritmos de aprendizaje automático y procesamiento de datos

## 5.1. El canal de preprocesamiento de la señal

Para pasar desde los datos puros que entrega una antena hasta la entrada de un algoritmo de aprendizaje automático, se debe seguir una secuencia estandarizada de transformación de datos:

Primero se toman las muestras de voltaje crudo en Fase y Cuadratura o las lecturas de potencia. Se aplican filtros digitales para eliminar el componente continuo de corriente y limpiar el ruido que producen las tarjetas electrónicas al calentarse.

Segundo, se aplica la Transformada de Fourier de Tiempo Corto. Este cálculo matemático toma la señal unidimensional en el tiempo y la va cortando en pequeñas ventanas solapadas, calculando la frecuencia de cada pedazo. El resultado final es una matriz de números en dos dimensiones (tiempo y frecuencia) que al graficarse forma una imagen en color llamada espectrograma.

Tercero, las imágenes de los espectrogramas se escalan a un tamaño fijo estándar (por ejemplo 128 por 128 píxeles) y sus valores de potencia en decibelios se normalizan para que todos los números queden apretados en una escala uniforme entre el cero y el uno.

Cuarto, para que los modelos no se aprendan de memoria las pocas fotos que tenemos, se aplica la aumentación de datos. Esto consiste en crear imágenes nuevas aplicando transformaciones matemáticas a las originales: sumando ruido sintético de fondo, desplazando las señales un poco hacia la derecha en el tiempo o bajándoles un poco el brillo (potencia).

## 5.2. Comparación entre usar datos tabulares o imágenes de espectrogramas

Existen dos filosofías completamente diferentes para alimentar los modelos:

Por un lado está el uso de características numéricas tabulares. Consiste en tomar la señal y calcularle valores estadísticos simples usando fórmulas matemáticas: el promedio de la potencia, el valor pico máximo, la varianza, el ancho de banda ocupado o la simetría de la onda. Estos números se acomodan en una tabla de Excel con filas y columnas. Las ventajas principales de esta opción son que los archivos pesan poquísimo, no se necesita una computadora potente con tarjeta gráfica para entrenar y modelos como el Bosque Aleatorio aprenden en un par de segundos. La desventaja es que dependemos de haber elegido buenas fórmulas a mano; si la interferencia tiene una forma rara que no se refleja en la varianza o en el promedio, el modelo no la va a ver.

Por otro lado está el uso directo de los espectrogramas como imágenes bidimensionales. Las ventajas principales son que no hay que inventar fórmulas a mano porque las redes neuronales convolucionales descubren por sí solas los bordes, las líneas diagonales y las texturas de la señal en la foto, logrando niveles de acierto muy superiores cuando la señal viene escondida dentro del ruido. Las desventajas son que se necesitan miles de fotos etiquetadas, el entrenamiento exige tarjetas gráficas avanzadas y los modelos se convierten en cajas negras donde es difícil explicar matemáticamente por qué la red tomó esa decisión.

## 5.3. Selección de algoritmos según el problema

Para características numéricas en tablas: el mejor candidato es el Bosque Aleatorio (*Random Forest*). Funciona creando decenas de árboles de decisión independientes que votan entre sí. Es sumamente estable, no se confunde fácil si faltan datos y logra porcentajes de acierto superiores al 96% en la clasificación de señales.

Para imágenes de espectrogramas: los modelos más adecuados son las Redes Neuronales Convolucionales ligeras y la arquitectura U-Net. Mientras que la red convolucional simple nos dice qué tipo de interferencia hay en toda la foto, la red U-Net nos devuelve una máscara dibujada encima de la imagen pintando exactamente qué píxeles en el tiempo y en la frecuencia están siendo aplastados por la interferencia.

Para series temporales de ocupación: las redes de Memoria a Corto y Largo Plazo (LSTM) son las indicadas para tomar la secuencia histórica de los estados del canal y predecir los números futuros.

Para detección de anomalías sin etiquetas: las redes Autoencoders Convolucionales son la mejor opción para aprender a reconstruir el espectro limpio y marcar cualquier desviación como una interferencia desconocida.

---

# Sección 6: Evaluación de modelos, comparación de resultados, problemas abiertos y propuesta experimental

## 6.1. Métricas de evaluación y cómo se aplican en la práctica

Para juzgar si nuestros modelos de aprendizaje automático funcionan bien o mal, no podemos usar simplemente el porcentaje de exactitud global (*Accuracy*). Si un canal de radio está desocupado el 95% del día y un modelo perezoso responde siempre "Libre" sin mirar nada, el modelo va a tener un 95% de exactitud global a pesar de ser completamente inútil.

Por eso, en este proyecto las métricas se deben calcular e interpretar de la siguiente manera sobre la Matriz de Confusión:

### 6.1.1. Cómo calculamos e interpretamos las métricas

Primero, construimos la Matriz de Confusión, que es una tabla de dos por dos donde cruzamos la realidad con lo que predijo el modelo:
- Verdaderos Positivos (VP): La señal estaba Ocupada y el modelo dijo Ocupada.
- Verdaderos Negativos (VN): La señal estaba Libre y el modelo dijo Libre.
- Falsos Positivos (FP): La señal estaba Libre pero el modelo se asustó y dijo Ocupada (Falsa Alarma).
- Falsos Negativos (FN): La señal estaba Ocupada pero el modelo no la vio y dijo Libre (Detección Fallida).

A partir de estos cuatro números, calculamos las métricas paso a paso:

La Precisión (*Precision*) se calcula dividiendo los Verdaderos Positivos entre la suma de Verdaderos Positivos más Falsos Positivos. Nos responde a la pregunta: de todas las veces que el modelo gritó "¡Hay una señal!", ¿en qué porcentaje estuvo realmente en lo cierto?

La Exhaustividad o Sensibilidad (*Recall* o $P_d$) se calcula dividiendo los Verdaderos Positivos entre la suma de Verdaderos Positivos más Falsos Negativos. Nos responde a la pregunta: de todas las señales reales que pasaron por el aire, ¿qué porcentaje logró cazar el modelo sin que se le escapara ninguna?

El F1-Score es la media armónica entre la Precisión y la Exhaustividad, y se calcula con la fórmula:
$$F1 = 2 \times \frac{\text{Precisión} \times \text{Exhaustividad}}{\text{Precisión} + \text{Exhaustividad}}$$
Esta es la métrica reina que debemos usar para comparar nuestros modelos, porque si el modelo falla gravemente en la Precisión o en la Exhaustividad, el F1-Score se desploma de inmediato.

La Probabilidad de Falsa Alarma ($P_{fa}$) se calcula dividiendo los Falsos Positivos entre la suma de Falsos Positivos más Verdaderos Negativos. Nos indica la tasa de veces que dijimos que el canal estaba ocupado por error estando vacío.

En el área de radiocomunicaciones, las normas internacionales exigen que cualquier modelo para operar en la vida real debe garantizar una Probabilidad de Detección ($P_d$) igual o mayor al 90% (0.90) manteniendo una Probabilidad de Falsa Alarma ($P_{fa}$) menor o igual al 10% (0.10).

## 6.2. Comparación de los resultados publicados en la literatura

Al revisar las publicaciones científicas más importantes de los últimos años, se observan los siguientes niveles de rendimiento alcanzados por otros investigadores:

En la detección de bloqueo intencional (*jamming*) en señales satelitales, investigaciones que usaron modelos clásicos de Bosque Aleatorio combinados con procesamiento de imágenes alcanzaron una exactitud media del 99% evaluando sobre registros reales.

En la búsqueda de fallas y anomalías en emisoras de radio en Frecuencia Modulada, los estudios que aplicaron redes Autoencoder Convolucionales sobre el dataset público de FM de Hungría lograron un F1-Score del 91.4%.

En el área de radioastronomía para limpiar datos de radiotelescopios, la red convolucional R-Net logró un Área Bajo la Curva de 0.91 cuando se usó aprendizaje por transferencia, comparado contra solo 0.67 cuando intentaron entrenar la red desde cero sin simulaciones previas.

En la detección de interferencias compuestas difíciles en señales de navegación, redes convolucionales ligeras como LcxNet lograron mantener un 90% de probabilidad de detección incluso cuando la potencia de la interferencia venía 20 decibelios por debajo del ruido de fondo (-20 dB JNR), consumiendo un 43% menos de parámetros que redes comerciales como ResNet-18.

En la predicción temporal de espacios en blanco a corto plazo, las redes neuronales recurrentes LSTM lograron niveles de acierto superiores al 88% prediciendo la ocupación de los siguientes minutos sobre trazas históricas de radio.

## 6.3. Problemas abiertos y dificultades no resueltas

A pesar de estos grandes números en los papeles, la literatura reconoce que existen cinco problemas severos que aún no tienen una solución definitiva:

La brecha entre la simulación y el mundo real: un modelo que saca 99% de acierto con datos sintéticos generados por una computadora suele caer a menos del 70% cuando se conecta a una antena real en la calle.

La trampa de las etiquetas etiquetadas con umbral: en la mayoría de los datasets, la etiqueta de "Canal Libre" no la puso un ser humano mirando la realidad, sino un programa que aplicó un umbral de energía simple. Si el programa original se equivocó, el modelo de aprendizaje automático va a aprender y repetir exactamente el mismo error.

El cambio progresivo de concepto (*Concept Drift*): los hábitos de consumo de radio cambian con el tiempo. Un modelo que aprendió perfectamente el tráfico de una ciudad durante el verano se vuelve inútil en invierno porque la gente cambia sus rutinas y horarios de uso.

La falta de generalización entre bandas: un modelo entrenado para detectar interferencias en la banda de posicionamiento por satélite (1.5 Gigahercios) es completamente incapaz de detectar interferencias en la banda de Wi-Fi (2.4 Gigahercios) debido a las diferencias en los anchos de banda y la física de propagación.

El costo computacional para dispositivos embebidos: los modelos de aprendizaje profundo más precisos necesitan tarjetas gráficas pesadas, lo que impide instalarlos dentro de sensores portátiles pequeños o radios económicas alimentadas por baterías.

## 6.4. Formulación de la propuesta experimental para nuestro proyecto

Tomando en cuenta todo lo aprendido en la literatura y considerando que **no contamos con hardware especializado de radiofrecuencia**, se propone desarrollar nuestro experimento académico dividido en dos líneas de trabajo independientes usando únicamente computadoras convencionales y conjuntos de datos públicos abiertos:

### 6.4.1. Línea A: Identificación y predicción de espacios en blanco

**Datos a utilizar:** El conjunto de datos real `LOG-a-TEC` (banda de 868 Megahercios) o el conjunto `AERPAW Packapalooza 2025` (datos de potencia reales).

**Paso a paso de la implementación:**
1. Tomamos las lecturas continuas de densidad espectral de potencia y aplicamos un umbral de energía adaptativo para generar nuestra columna de etiquetas binarias (0 para Libre, 1 para Ocupado).
2. Armamos vectores de entrada donde cada fila contiene las potencias pasadas de los últimos $N$ pasos de tiempo.
3. Dividimos los datos de forma estrictamente cronológica: el primer 70% de las horas del día para entrenar, el siguiente 15% para validar hiperparámetros y el último 15% del día para la prueba final. **Bajo ninguna circunstancia debemos mezclar los datos al azar**, porque las mediciones de radio consecutivas se parecen mucho entre sí y mezclar el tiempo causaría una trampa de filtración de datos (*data leakage*) que inflaría falsamente los resultados.
4. Entrenamos y comparamos tres modelos:
   - Una línea base tradicional: Detección de energía por umbral fijo.
   - Un modelo de aprendizaje automático clásico: **Bosque Aleatorio** (*Random Forest*).
   - Un modelo de serie temporal: **Red Neuronal Recurrente LSTM**.
5. Calculamos y graficamos la Precisión, el F1-Score, la Probabilidad de Detección ($P_d$) y la Probabilidad de Falsa Alarma ($P_{fa}$) para demostrar cuál modelo predice mejor los huecos libres del futuro.

### 6.4.2. Línea B: Detección y clasificación de interferencias

**Datos a utilizar:** El conjunto de datos real `Dataset de Laboratorio para GNSS de Fraunhofer` (subconjunto FIOT descargado mediante Git LFS) o una muestra seleccionada del dataset `Indoor Jamming Scenarios`.

**Paso a paso de la implementación:**
1. Extraemos las muestras de Fase y Cuadratura o las matrices de densidad espectral y las organizamos según sus etiquetas de clase conocidas (Sin interferencia, Interferencia de tono puro, Interferencia de barrido *chirp*, Interferencia de pulso, Ruido blanco).
2. Para la rama de características tabulares: calculamos a mano por código la media, varianza, valor máximo y curtosis de la potencia de cada muestra y armamos un archivo de formato CSV.
3. Para la rama de imágenes: aplicamos la Transformada de Fourier de Tiempo Corto (STFT) a cada muestra y guardamos espectrogramas en formato de imagen de 128 por 128 píxeles.
4. Dividimos los datos reservando un 20% de los archivos para la prueba final sin tocar.
5. Entrenamos y comparamos dos enfoques:
   - Enfoque Tabular: **Bosque Aleatorio** (*Random Forest*) sobre el archivo CSV de características calculadas.
   - Enfoque Visión: **Red Neuronal Convolucional (CNN) Ligera** (de 3 capas convolucionales) sobre las imágenes de los espectrogramas.
6. Evaluamos el rendimiento construyendo la Matriz de Confusión completa para ver exactamente qué clase de interferencia se confunde con cuál, calculamos el F1-Score macro para no favorecer a ninguna clase y medimos el tiempo promedio en milisegundos que tarda cada modelo en clasificar una muestra en la computadora.

---

# Conclusión General del Estado del Arte

Esta revisión bibliográfica del estado del arte nos permite sacar tres conclusiones fundamentales para nuestro trabajo:

En primer lugar, hemos verificado que el monitoreo del espectro mediante aprendizaje automático ha dejado atrás las reglas estáticas rígidas para adoptar modelos adaptativos. Los modelos clásicos tabulares como el Bosque Aleatorio son imbatibles en rapidez y bajo consumo de cómputo sobre tablas de atributos, mientras que las redes convolucionales sobre espectrogramas son la herramienta más potente para detectar firmas de interferencias complejas dentro del ruido.

En segundo lugar, aprendimos que el procesamiento adecuado de los datos (transformar señales crudas a imágenes de tiempo-frecuencia con la Transformada de Fourier de Tiempo Corto, normalizar potencias y respetar una división temporal estricta sin mezclar el pasado con el futuro) es más importante para el éxito final que el solo hecho de usar una red neuronal gigante.

En tercer lugar, demostramos que no necesitamos comprar antenas ni receptores costosos para realizar un proyecto científico de excelente calidad. La ruta trazada con la **Línea A** (usando datos reales de `LOG-a-TEC` para predecir espacios en blanco con Bosque Aleatorio y redes LSTM) y la **Línea B** (usando datos reales de `Fraunhofer` para clasificar interferencias con Bosque Aleatorio y redes convolucionales) nos permite construir un experimento completo, con métricas estrictas y totalmente realizable dentro del alcance universitario.

