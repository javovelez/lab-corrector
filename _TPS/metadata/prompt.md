Actúa como un asistente de programación experto en automatización. A continuación te detallo un flujo que debes ejecutar mediante un script de Python. Te etiquetaré dos archivos: el notebook inicial que recibe un alumno y el notebook de su solución oficial.

Tienes que generar dos nuevos archivos en formato Markdown que deberán guardarse en el mismo directorio donde se ubique el notebook de la solución. Tu trabajo es interpretar el JSON interno de los notebooks y aplicar estrictamente esta lógica:

1. Archivo de Resolución: {Nombre_Laboratorio_Solucion}.md Basándote en el notebook de SOLUCIÓN, genera un documento Markdown aplicando estas acciones sobre cada celda:

Celdas de resolución (code): Conserva el código envolviéndolo en bloques formatting python . Deben estar sin outputs.
Celdas de "setup" (code): Elimina/ignora cualquier celda orientada a configurar el entorno (típicamente contienen el flag de descargas ej. !gdown, carga de importaciones base, revisiones de versión y funciones predefinidas de testing).
Enunciados (markdown): Si la celda es la cabecera de un ejercicio (suele comenzar textualmente con "### Ejercicio"), borra todo el contenido escrito y escribe allí ÚNICAMENTE su "id" de celda (ej: id-de-la-celda).
Respuestas de Análisis (markdown): Si la celda detalla la solución teórica (suele incluir "Respuesta a la pregunta de análisis:"), DEBE mantenerse el texto original íntegro.
Resto de texto (markdown): Debes omitir e ignorar introducciones teóricas, títulos de seccionado general, notas de corrección y explícitamente debes ignorar los enunciados de las "Preguntas de análisis".
2. Archivo de Reporte: {Nombre_Laboratorio}_eliminados.md Basándote ahora en las celdas del notebook ORIGINAL del alumno, crea un reporte en estilo lista (bullets) que contenga solo los "id" (- {cell_id}) de las celdas que fueron omitidas o reemplazadas. La lógica clasificatoria para extraer estos IDs es:

INCLUIR los IDs de todas las celdas de código descartadas por ser de setup.
INCLUIR los IDs de todas las celdas markdown correspondientes a introducciones teóricas, subtítulos de relleno, footers y de los apartados de enunciados de ejercicios (aquellos que se reemplazaron solo por su ID).
EXCLUIR RIGUROSAMENTE los IDs de las celdas de "Preguntas de Análisis". Ningún ID de una pregunta teórica de análisis debe estar en este documento.
Ejecuta este flujo automáticamente por mí de un solo intento para estos dos archivos en específico: Original: @[_TPS/Laboratorios/Laboratorio_1b.ipynb] Solución: @[_TPS/Soluciones/Laboratorio_1b_Solucion.ipynb]