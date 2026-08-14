# Corpus de demostración — Retrieval Híbrido

Documentos de ejemplo sobre sistemas de IA en producción. Contenido público original (sin secretos ni datos privados), creado para que el harness de evals sea reproducible offline. Varios documentos comparten vocabulario a propósito (distractores) para que la búsqueda híbrida demuestre su valor frente a la puramente léxica.

# Generación aumentada por recuperación

Los sistemas de generación aumentada por recuperación anclan las respuestas de un modelo de lenguaje a un conjunto de documentos previamente indexados. En lugar de confiar solo en los parámetros del modelo, el sistema primero busca fragmentos relevantes en una base de conocimiento y luego compone la respuesta a partir de ellos. Esto reduce las respuestas inventadas y permite citar la fuente exacta de cada afirmación, algo imprescindible en entornos corporativos con datos privados o normativos.

# Búsqueda léxica con BM25

BM25 es un algoritmo de ranking clásico basado en la frecuencia de términos. Asigna a cada documento una puntuación que depende de cuántas veces aparece cada palabra de la consulta, ponderada por la rareza del término en el corpus completo. Funciona especialmente bien cuando la consulta contiene acrónimos, nombres propios, identificadores o jerga técnica exacta. Su debilidad aparece cuando la consulta usa sinónimos: un texto que habla de "vehículos" no se relaciona con uno que pregunta por "coches".

# Búsqueda semántica con embeddings

La búsqueda semántica representa cada documento como un vector numérico generado por un modelo de lenguaje. La similitud entre la consulta y un documento se calcula como el coseno entre sus vectores. Así, dos textos que usan palabras distintas pero significan lo mismo quedan cerca en el espacio vectorial. Su punto débil es el opuesto a BM25: los términos poco frecuentes pueden perderse si no están bien representados en el espacio semántico del modelo.

# Fusión con RRF (Reciprocal Rank Fusion)

RRF combina varias listas de resultados sin necesidad de normalizar puntuaciones de naturaleza distinta. Para cada documento se suman las contribuciones de cada lista, donde la contribución es el inverso de su posición en dicha lista más una constante k, habitualmente 60. Un documento que aparece en segundo lugar en la lista léxica y en quinto en la semántica acumula puntos en ambos rankings. Esta técnica es la base de la búsqueda híbrida en bases de datos vectoriales modernas.

# Reranking con cross-encoders

Los retrievers de primera fase devuelven candidatos baratos pero imprecisos. Un cross-encoder procesa el par consulta-documento a la vez y devuelve una puntuación de relevancia mucho más fiel al significado completo. Como es computacionalmente caro, se aplica solo a los pocos candidatos mejor clasificados por la fase anterior. La combinación de una fase de recuperación amplia y una fase de reranking estrecha es el patrón estándar en producción.

# Evaluación de un sistema RAG

Evaluar un sistema RAG exige separar la calidad de la recuperación de la calidad de la generación. Para la recuperación se usan métricas de ranking como recall en los k primeros resultados, MRR y NDCG, comparando contra un conjunto de consultas con documentos relevantes marcados a mano. Para la generación se mide la fidelidad de las afirmaciones con respecto a la fuente recuperada. Los equipos que automatizan esta evaluación pueden comparar cada cambio del sistema con números antes y después.

# Métricas de ranking: recall, MRR y NDCG

El recall en k indica la proporción de documentos relevantes que aparecen entre los k primeros resultados. El MRR premia que el primer resultado relevante aparezca lo antes posible en la lista. El NDCG pondera la posición de todos los resultados relevantes con un descuento logarítmico, de modo que un acierto en primera posición vale más que uno en novena. Juntas dan una imagen completa de la utilidad de una lista de resultados.

# Agentes con estado en producción

Un agente autónomo con memoria de estado puede interrumpir su ejecución para pedir confirmación humana y luego reanudar exactamente donde quedó. Los frameworks modernos permiten persistir el grafo de ejecución en una base de datos, de forma que un fallo en un nodo no reinicia todo el flujo. Este patrón de parada y reanudación es clave para tareas con impacto económico: aprobaciones de pago, revisiones de código o despliegues. Los bucles de acción fallidos se registran para depuración y mejora continua.

# Streaming de respuestas con HTTP

Entregar respuestas largas de un modelo como un único bloque obliga al usuario a esperar decenas de segundos. La alternativa es el streaming por fragmentos: el servidor envía cada fragmento en cuanto lo produce, y el cliente lo renderiza progresivamente. La mecánica estándar usa respuestas con eventos enviados por el servidor, que mantienen la conexión abierta mientras el generador de texto produce tokens. Esto mejora la sensación de velocidad y permite cancelar la generación en cualquier momento sin matar el proceso.

# Observabilidad de modelos de lenguaje

Cuando un sistema de lenguaje falla en producción, la causa rara vez es un error de código; suele ser una salida de baja calidad que pasa desapercibida. La observabilidad registra cada llamada al modelo con su entrada, salida, latencia, coste y metadatos de contexto. Los paneles de coste por usuario y latencia por modelo permiten detectar deriva de calidad antes de que afecte a los usuarios finales. Comparar el comportamiento actual contra un conjunto de ejemplos dorados es la forma más robusta de alertar.

# Protocolo MCP para integrar herramientas

El protocolo de contexto de modelo define una forma estándar de conectar modelos de lenguaje con herramientas externas. Cada herramienta se declara con un esquema formal que describe sus parámetros y el modelo puede invocarla cuando la tarea lo requiere. Las respuestas estructuradas con formato validado evitan que el modelo devuelva JSON improvisado. La seguridad del protocolo exige validar siempre la entrada que llega de una herramienta y limitar los permisos al mínimo necesario.

# Bases de datos vectoriales en producción

Las bases de datos vectoriales almacenan los embeddings junto con filtros de metadatos y ofrecen búsqueda por similitud escalable. La extensión pgvector añade este tipo de búsqueda a PostgreSQL, permitiendo que los datos operacionales y los vectores vivan en el mismo sistema transaccional. Los índices de vecinos aproximados sacrifican una fracción de exactitud a cambio de tiempos de consulta constantes incluso con millones de vectores.

# Seguridad en aplicaciones de IA

Las aplicaciones de lenguaje introducen una superficie de ataque nueva: la inyección de instrucciones en el texto procesado. Un atacante puede intentar redirigir el comportamiento del modelo incrustando órdenes ocultas en un documento recuperado. La defensa combina saneamiento de entrada, aislamiento de herramientas con privilegios mínimos y supervisión humana en acciones con impacto. Auditar periódicamente qué hace el sistema con entradas maliciosas es parte del ciclo de vida del producto.

# Inyección de instrucciones en sistemas aumentados

La inyección de instrucciones ocurre cuando contenido no confiable, como un correo o una página web, consigue alterar las órdenes dadas al modelo. En un sistema de recuperación, un documento malicioso indexado puede inyectar comandos cuando es recuperado y pasado al generador. Las mitigaciones incluyen separar claramente las instrucciones del sistema del contenido no confiable, no otorgar herramientas de alto privilegio al modelo sin confirmación, y entrenar al equipo para detectar comportamientos anómalos de salida.

# Similitud coseno en espacios vectoriales

La medida más habitual de cercanía entre dos embeddings es el coseno del ángulo que forman sus vectores: 1 significa idéntica dirección, 0 significa ortogonalidad total. Cuando los vectores están normalizados a norma unitaria, el coseno se convierte en un simple producto escalar, lo que permite usar multiplicación de matrices para calcular la similitud de una consulta contra todo el corpus en un solo paso. Esta operación es el núcleo de cualquier índice denso.

# Caché semántica de respuestas

Repetir la generación para consultas casi idénticas desperdicia tiempo y dinero. Una caché semántica guarda las respuestas asociadas a los embeddings de las consultas ya resueltas: si una consulta nueva cae por encima de un umbral de similitud con alguna guardada, se devuelve la respuesta en caché sin llamar al modelo. El ahorro de coste llega a dobles dígitos en cargas repetitivas de soporte, a costa de una latencia mínima extra en la verificación de similitud.

# Hallucination en modelos de lenguaje

Los modelos generan texto plausible sin garantía de veracidad. Cuando una afirmación no se apoya en ninguna fuente, se habla de alucinación: el modelo "rellena" la respuesta con conocimiento que parece razonable pero puede ser falso. Las estrategias de mitigación más eficaces combinan recuperación de fuentes, instrucciones que obligan a citar, y revisores de fidelidad que comparan cada afirmación contra el fragmento citado antes de entregar la respuesta al usuario.

# Autenticación con tokens JWT

Los tokens de acceso firmados se usan para transmitir la identidad del usuario entre servicios sin mantener sesiones en el servidor. Un token contiene cabecera, carga útil y firma; la carga útil suele incluir el identificador del usuario y una fecha de expiración. Verificar la firma con la clave pública del emisor y comprobar la expiración son los dos chequeos mínimos antes de confiar en cualquier token recibido.

# Auditoría de cabeceras HTTP de seguridad

Las cabeceras de seguridad endurecen el navegador frente a ataques comunes: Content-Security-Policy restringe los orígenes desde los que se cargan recursos, Strict-Transport-Security fuerza conexiones HTTPS, y X-Frame-Options impide incrustar la página en iframes de terceros. Auditar un sitio consiste en solicitar la respuesta y comprobar la presencia y el valor correcto de estas cabeceras. La ausencia de una de ellas es una vulnerabilidad de baja severidad pero fácil de corregir.

# Inspección de puertos en red

Un escáner de puertos intenta conectar a direcciones y puertos concretos para descubrir qué servicios se están ejecutando. El resultado es una lista de puertos abiertos que constituye el primer mapa de ataque de una infraestructura. En pruebas autorizadas, el escaneo se acompaña de identificación de versión del servicio para cruzar con bases de datos de vulnerabilidades conocidas. La velocidad se controla para no saturar el objetivo, y el alcance siempre se limita a rangos autorizados.

# Enumeración de subdominios

Antes de atacar un objetivo, conviene cartografiar su superficie: la enumeración de subdominios descubre nombres de host asociados al dominio principal mediante consultas al sistema de nombres, diccionarios de nombres comunes y fuentes de certificados públicos. Cada subdominio encontrado es un nuevo servicio potencialmente desplegado con configuración distinta y, a menudo, menos endurecida. Es la primera fase de cualquier ejercicio de reconocimiento.
