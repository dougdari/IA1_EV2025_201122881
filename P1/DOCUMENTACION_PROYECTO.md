# Documentación del Sistema de Diagnóstico Médico UniMatch

## Índice

1. [Visión General del Proyecto](#visión-general-del-proyecto)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Documentación del Backend](#documentación-del-backend)
4. [Documentación del Frontend](#documentación-del-frontend)
5. [Flujo de Datos](#flujo-de-datos)
6. [Instalación y Configuración](#instalación-y-configuración)
7. [Casos de Uso](#casos-de-uso)

---

## Visión General del Proyecto

UniMatch es un sistema inteligente de análisis médico que utiliza múltiples técnicas de inteligencia artificial para evaluar síntomas de pacientes, detectar enfermedades y recomendar medicamentos apropiados evitando contraindicaciones.

### Objetivo Principal

Proporcionar una herramienta de asistencia médica que analice texto en lenguaje natural sobre síntomas y condiciones del paciente, y genere recomendaciones basadas en:

- Análisis de procesamiento de lenguaje natural (NLP)
- Clasificación mediante aprendizaje automático
- Razonamiento lógico mediante sistemas expertos

### Tecnologías Principales

**Backend:**
- Go (Golang) - Lenguaje principal del servidor
- Fiber - Framework web de alto rendimiento
- Gonum - Biblioteca matemática para machine learning
- Golog - Motor de inferencia Prolog
- HuggingFace API - Modelo de lenguaje médico

**Frontend:**
- JavaScript moderno (ES6+)
- Vite - Herramienta de construcción y desarrollo
- HTML5 y CSS3
- Arquitectura modular basada en componentes

---

## Arquitectura del Sistema

### Estructura de Directorios

```
P1/
├── backend/
│   ├── api.go                    # Servidor principal y endpoints
│   ├── train.go                  # Funciones de entrenamiento
│   ├── train_main.go             # Punto de entrada para entrenamiento
│   ├── go.mod                    # Dependencias del proyecto
│   ├── rpa.ps1                   # Script de automatización
│   ├── algorithms/
│   │   ├── softmax.go            # Implementación del modelo Softmax
│   │   └── bronco_dataset.csv    # Dataset de entrenamiento
│   ├── prolog/
│   │   ├── conocimiento.pl       # Base de conocimiento médico
│   │   └── inferencias.pl        # Reglas de inferencia
│   └── weights/
│       ├── softmax_model.json    # Modelo entrenado serializado
│       └── *.csv                 # Datos de entrenamiento
│
├── Frontend/
│   └── chat-frontend/
│       ├── index.html            # Página principal
│       ├── package.json          # Dependencias
│       └── src/
│           ├── main.js           # Punto de entrada
│           ├── style.css         # Estilos globales
│           ├── api/
│           │   └── chatApi.js    # Cliente HTTP
│           └── components/
│               └── Chat.js       # Componente de interfaz de chat
│
└── docs/
    ├── manual_tecnico.md         # Documentación técnica
    └── manual_usuario.md         # Manual de usuario
```

### Componentes del Sistema

#### 1. Servidor Backend (Go)

El backend está construido con Go y utiliza el framework Fiber para proporcionar una API REST de alto rendimiento. Gestiona toda la lógica de negocio del sistema.

#### 2. Módulo de Procesamiento de Texto

Analiza el texto del paciente extrayendo características relevantes como:
- Número de síntomas mencionados
- Presencia de condiciones crónicas
- Indicadores de alerta (red flags) como dolor de pecho o dificultad respiratoria

#### 3. Integración con HuggingFace

Se conecta con el modelo de lenguaje biomédico en español `bsc-bio-ehr-es` para obtener probabilidades de enfermedades basadas en el contexto médico del texto.

#### 4. Modelo de Machine Learning (Softmax)

Implementación personalizada de regresión softmax (regresión logística multinomial) que clasifica el nivel de urgencia y tipo de enfermedad basándose en:
- Probabilidades del modelo de lenguaje
- Características extraídas del texto
- Indicadores de riesgo

#### 5. Sistema Experto (Prolog)

Motor de inferencia lógica que contiene conocimiento médico sobre contraindicaciones de medicamentos basándose en:
- Tipo de enfermedad
- Nivel de urgencia
- Condiciones crónicas
- Síntomas de alerta

#### 6. Cliente Frontend

Interfaz web moderna construida con Vite que proporciona una experiencia de chat para la interacción con el sistema.

---

## Documentación del Backend

### Archivo: api.go

Este es el archivo principal del backend que contiene toda la lógica del servidor y el procesamiento de diagnósticos.

#### Estructuras de Datos Principales

**MedicamentoRecomendado**
```go
type MedicamentoRecomendado struct {
    Urgencia    string   // Nivel de urgencia (baja/mediana/alta)
    Enfermedad  string   // Tipo de enfermedad
    Cronica     string   // Si es crónica (cronica_si/cronica_no)
    Pecho       string   // Dolor de pecho (pecho_si/pecho_no)
    Respiracion string   // Dificultad respiratoria (resp_si/resp_no)
    Medicamento string   // Nombre del medicamento
    Match       float64  // Porcentaje de coincidencia con el caso
}
```

**DiagnosticoRequest**
```go
type DiagnosticoRequest struct {
    Texto string `json:"texto"` // Texto del paciente describiendo síntomas
}
```

**DiagnosticoResponse**
```go
type DiagnosticoResponse struct {
    EnfermedadDetectada       string                   // Enfermedad identificada
    NivelUrgencia             string                   // baja/mediana/alta
    ProbabilidadesHuggingFace map[string]float64       // Probabilidades del modelo NLP
    ClaseSoftmax              int                      // Clase predicha por el modelo
    MedicamentosEvaluados     []MedicamentoRecomendado // Lista de medicamentos
    TotalContraindicados      int                      // Cantidad de contraindicaciones
    Advertencias              []string                 // Mensajes de alerta
    TextoRecibido             string                   // Texto original del usuario
}
```

**VectorEntrada**
```go
type VectorEntrada struct {
    // Probabilidades de enfermedades del modelo NLP
    a_asma         float32
    a_bronquitis   float32
    a_enfisema     float32
    a_apnea        float32
    a_fibromialgia float32
    a_migranas     float32
    a_reflujo      float32
    
    // Características extraídas del texto
    n_sintomas          int    // Conteo de síntomas mencionados
    n_cronicas          int    // Conteo de condiciones crónicas
    redflag_pecho       bool   // Indica dolor de pecho
    redflag_respiracion bool   // Indica dificultad respiratoria severa
    tiene_cronicas      bool   // Si hay condiciones crónicas
}
```

#### Mapeos de Clasificación

El sistema utiliza un mapeo directo entre las clases numéricas del modelo y diagnósticos médicos:

```go
var clasificacionMedica = map[int]struct {
    Urgencia   string
    Enfermedad string
    Cronica    string
}{
    0: {"baja", "ninguna", "cronica_no"},
    1: {"mediana", "asma", "cronica_si"},
    2: {"alta", "bronquitis", "cronica_si"},
    3: {"alta", "enfisema", "cronica_si"},
    4: {"mediana", "apnea", "cronica_si"},
    5: {"baja", "fibromialgia", "cronica_si"},
    6: {"baja", "migranas", "cronica_si"},
    7: {"baja", "reflujo", "cronica_si"},
}
```

#### Palabras Clave para Análisis de Texto

El sistema busca palabras clave específicas en el texto del paciente:

**Síntomas:**
- pecho, tos, flema, silbido
- falta de aire, ahogo, dificultad para respirar
- opresión, dolor al respirar
- cansancio, fatiga, sibilancias
- esputo, mucosidad

**Condiciones Crónicas:**
- asma, epoc, bronquitis crónica
- fibrosis pulmonar, enfisema, apnea
- términos temporales: años de, desde hace

**Red Flags - Dolor de Pecho:**
- dolor de pecho, opresión en el pecho
- presión en el pecho, pecho apretado
- dolor torácico, dolor intenso en el pecho

**Red Flags - Respiración Severa:**
- no puedo respirar, falta de aire severa
- ahogo, dificultad extrema
- labios azules, cianosis, me ahogo

#### Funciones Principales

**1. analizarTexto(texto string) FeaturesTexto**

Analiza el texto del paciente y extrae características numéricas y booleanas.

Proceso:
1. Convierte el texto a minúsculas para búsqueda case-insensitive
2. Cuenta ocurrencias de palabras clave de síntomas
3. Detecta menciones de condiciones crónicas
4. Identifica red flags de pecho y respiración
5. Retorna un objeto FeaturesTexto con todas las características

**2. llamarHuggingFace(texto string) (map[string]float64, error)**

Se conecta con la API de HuggingFace para obtener probabilidades de enfermedades.

Proceso:
1. Lee el token de autenticación desde la variable de entorno HF_TOKEN
2. Añade el token `<mask>` al texto para que el modelo prediga enfermedades
3. Envía una petición POST al endpoint del modelo bsc-bio-ehr-es
4. Parsea la respuesta JSON para extraer probabilidades por enfermedad
5. Retorna un mapa con enfermedad:probabilidad

**3. obtenerMedicamentosContraindicados(...)**

Consulta la base de conocimiento Prolog para obtener medicamentos contraindicados.

Parámetros:
- m: Máquina Prolog
- urgencia: Nivel de urgencia detectado
- enfermedad: Enfermedad diagnosticada
- cronica: Estado crónico
- pecho: Presencia de dolor de pecho
- respiracion: Dificultad respiratoria

Proceso:
1. Construye una consulta Prolog con las variables proporcionadas
2. Ejecuta la consulta en el motor Prolog
3. Por cada solución encontrada, calcula un porcentaje de coincidencia
4. El porcentaje considera múltiples factores ponderados
5. Retorna lista de medicamentos con su nivel de match

**4. procesarDiagnostico(req DiagnosticoRequest) (*DiagnosticoResponse, error)**

Pipeline completo de diagnóstico que integra todos los componentes.

Flujo de ejecución:
1. **Análisis de texto:** Extrae características del texto del paciente
2. **HuggingFace:** Obtiene probabilidades de enfermedades mediante NLP
3. **Construcción del vector:** Combina probabilidades y características
4. **Clasificación Softmax:** Predice clase de urgencia/enfermedad
5. **Mapeo médico:** Traduce la clase a diagnóstico legible
6. **Consulta Prolog:** Obtiene medicamentos contraindicados
7. **Generación de advertencias:** Crea mensajes de alerta según el caso
8. **Respuesta:** Retorna objeto completo con toda la información

#### Endpoints del API

**GET /**

Endpoint de información que describe el servicio.

Respuesta:
```json
{
  "servicio": "UniMatch Medical Diagnosis API",
  "version": "3.0",
  "estado": "activo",
  "descripcion": "Sistema que evalua medicamentos y detecta contraindicaciones",
  "flujo": [
    "1. Analisis de texto (sintomas y red flags)",
    "2. HuggingFace (probabilidades de enfermedades)",
    "3. Softmax (clasificacion final)",
    "4. Prolog (entrega todos los medicamentos)",
    "5. Evaluacion (marca cuales estan contraindicados)"
  ]
}
```

**POST /diagnostico**

Endpoint principal que procesa el diagnóstico completo.

Entrada:
```json
{
  "texto": "Tengo tos seca desde hace 3 semanas, me duele el pecho al respirar"
}
```

Salida:
```json
{
  "enfermedad_detectada": "bronquitis",
  "nivel_urgencia": "alta",
  "probabilidades_huggingface": {
    "asma": 0.12,
    "bronquitis": 0.65,
    "enfisema": 0.08
  },
  "clase_softmax": 2,
  "medicamentos_evaluados": [
    {
      "Urgencia": "alta",
      "Enfermedad": "bronquitis",
      "Cronica": "cronica_si",
      "Pecho": "pecho_si",
      "Respiracion": "resp_si",
      "Medicamento": "sedantes_fuertes",
      "Match": 85.5
    }
  ],
  "total_contraindicados": 3,
  "advertencias": [
    "Diagnostico: bronquitis",
    "URGENCIA ALTA: Se recomienda atencion medica inmediata",
    "RED FLAG: Dolor o presion en el pecho detectado"
  ],
  "texto_recibido": "Tengo tos seca desde hace 3 semanas, me duele el pecho al respirar"
}
```

**POST /softmax/train**

Entrena un nuevo modelo Softmax con datos proporcionados.

Entrada:
```json
{
  "x": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  "y": [0, 1],
  "lr": 0.1,
  "n_iter": 2000,
  "reg_lambda": 0.001
}
```

**POST /softmax/predict**

Realiza predicciones con el modelo Softmax entrenado.

Entrada:
```json
{
  "x": [[1.5, 2.5, 3.5]]
}
```

Salida:
```json
{
  "y_pred": [0],
  "probs": [[0.85, 0.10, 0.05]]
}
```

### Archivo: algorithms/softmax.go

Implementación completa del algoritmo de regresión softmax (regresión logística multinomial).

#### Estructura Principal

```go
type SoftmaxRegression struct {
    W         *mat.Dense    // Matriz de pesos (nFeatures x nClasses)
    B         *mat.VecDense // Vector de sesgos (nClasses)
    Lr        float64       // Tasa de aprendizaje
    NIter     int           // Número de iteraciones
    RegLambda float64       // Fuerza de regularización L2
    LossHistory []float64   // Historial de pérdida por iteración
}
```

#### Funcionamiento del Modelo

**Regresión Softmax**

La regresión softmax es una generalización de la regresión logística para múltiples clases. Calcula probabilidades para cada clase posible.

**Función Forward**

Realiza la propagación hacia adelante:
1. Calcula scores = X·W + b
2. Aplica softmax a los scores para obtener probabilidades
3. Retorna tanto los scores como las probabilidades

**Función Softmax**

Transforma los scores en probabilidades que suman 1:
- Resta el máximo por fila para estabilidad numérica
- Calcula exponenciales
- Normaliza dividiendo por la suma

**Entrenamiento (Fit)**

Proceso de entrenamiento mediante descenso de gradiente:

1. **Inicialización:**
   - Determina el número de clases del dataset
   - Inicializa pesos W con valores aleatorios pequeños
   - Inicializa sesgos B en ceros

2. **One-Hot Encoding:**
   - Convierte las etiquetas a formato one-hot
   - Ejemplo: clase 2 de 4 clases → [0, 0, 1, 0]

3. **Iteraciones de Entrenamiento:**
   - Para cada iteración:
     - Calcula probabilidades mediante forward pass
     - Calcula pérdida cross-entropy con regularización L2
     - Calcula gradientes de W y B
     - Actualiza parámetros: W = W - lr·∇W, B = B - lr·∇B

4. **Regularización L2:**
   - Penaliza pesos grandes para evitar overfitting
   - Añade 0.5·λ·||W||² a la función de pérdida
   - Añade λ·W al gradiente

**Predicción (Predict)**

Para hacer predicciones:
1. Calcula probabilidades con PredictProba
2. Selecciona la clase con mayor probabilidad (argmax)
3. Retorna el índice de la clase predicha

**Métricas (Accuracy)**

Calcula la precisión del modelo:
- Compara predicciones con etiquetas reales
- Retorna la fracción de predicciones correctas

**Persistencia del Modelo**

El modelo se puede guardar y cargar desde disco:

**SaveToFile:**
- Serializa W, B y hiperparámetros a JSON
- Guarda en el path especificado

**LoadSoftmaxRegression:**
- Lee el archivo JSON
- Reconstruye las matrices W y B
- Retorna el modelo listo para usar

### Archivo: prolog/conocimiento.pl

Base de conocimiento médico que define contraindicaciones de medicamentos.

#### Estructura de las Reglas

Cada regla sigue el formato:
```prolog
medicamento_contraindicado(Urgencia, Enfermedad, Cronica, Pecho, Respiracion, Medicamento).
```

#### Categorías de Contraindicaciones

**1. Crisis Respiratoria Severa**

En casos de alta urgencia con problemas respiratorios:
- Evitar sedantes fuertes (depresión respiratoria)
- Evitar opioides fuertes (suprimen el reflejo respiratorio)
- Evitar benzodiacepinas en apnea

**2. Enfermedad Pulmonar Crónica**

Para condiciones crónicas respiratorias:
- Evitar betabloqueantes no selectivos (broncoconstricción)
- Evitar antitusivos opioides en enfisema
- Evitar medicamentos que agraven la función pulmonar

**3. Dolor de Pecho con Sospecha Pulmonar**

Cuando hay dolor torácico:
- Evitar AINEs en altas dosis hasta descartar causas cardíacas
- Precaución con medicamentos que enmascaren síntomas graves

**4. Apnea del Sueño**

Para pacientes con apnea:
- Evitar hipnóticos fuertes (empeoran la apnea)
- Evitar opioides (deprimen el centro respiratorio)
- Evitar cualquier depresor del sistema nervioso central

**5. Condiciones con Baja Urgencia**

En fibromialgia y migrañas:
- Evitar corticoides sistémicos prolongados (efectos secundarios)
- Evitar triptanes con antecedentes pulmonares

**6. Reflujo con Dolor de Pecho**

Para reflujo gastroesofágico:
- Evitar AINEs gastrolesivos (agravan síntomas digestivos)
- Especial cuidado si hay dolor torácico

#### Ejemplos de Reglas

```prolog
% Alta urgencia + asma + crónica + dolor de pecho + dificultad respiratoria
medicamento_contraindicado(alta, asma, cronica_si, pecho_si, resp_si, sedantes_fuertes).

% Mediana urgencia + asma + no crónica + dolor de pecho
medicamento_contraindicado(mediana, asma, cronica_no, pecho_si, resp_no, aines_altas_dosis).

% Baja urgencia + apnea + crónica + sin dolor de pecho
medicamento_contraindicado(baja, apnea, cronica_si, pecho_no, resp_no, hipnoticos_fuertes).
```

### Archivo: prolog/inferencias.pl

Contiene las reglas de inferencia para consultar la base de conocimiento.

```prolog
% Regla de recomendación
recomendar_medicamento(Urg, Enf, Cron, Pecho, Resp, Med) :-
    medicamento_contraindicado(Urg, Enf, Cron, Pecho, Resp, Med).
```

Esta regla simple permite consultar todos los medicamentos contraindicados que coincidan con los parámetros proporcionados.

### Archivo: train.go

Contiene funciones auxiliares para entrenar el modelo Softmax.

**Función TrainSoftmax**

Ejemplo de entrenamiento con un dataset de prueba:
1. Crea un dataset sintético en 2D con 3 clases
2. Inicializa el modelo con hiperparámetros
3. Entrena el modelo con el método Fit
4. Calcula la precisión en el conjunto de entrenamiento
5. Genera predicciones en puntos nuevos
6. Exporta los resultados a archivos CSV para visualización

**Función exportPointsCSV**

Exporta datos de entrenamiento y predicciones a formato CSV:
- Coordenadas de los puntos (x1, x2)
- Etiqueta verdadera (y_true)
- Clase predicha (y_pred)
- Probabilidades por cada clase (p0, p1, p2, ...)

Este formato permite visualizar fácilmente los resultados en Python, R o Excel.

---

## Documentación del Frontend

### Archivo: index.html

Página HTML simple que sirve como punto de entrada:
- Define la estructura básica del documento
- Contiene un div con id "app" donde se monta la aplicación
- Carga el módulo principal main.js

### Archivo: src/main.js

Punto de entrada del frontend:
```javascript
import './style.css'
import { Chat } from './components/Chat.js'

Chat(document.getElementById('app'))
```

Importa los estilos globales y el componente Chat, luego lo monta en el elemento app.

### Archivo: src/style.css

Define los estilos globales de la aplicación:
- Reset de márgenes y padding
- Estilos del body centrado
- Paleta de colores del tema oscuro

### Archivo: src/api/chatApi.js

Cliente HTTP que se comunica con el backend.

**Función sendMessage(texto)**

Envía el texto del paciente al backend:
```javascript
export async function sendMessage(texto) {
  const res = await fetch('http://localhost:8080/diagnostico', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texto })
  })
  
  if (!res.ok) throw new Error('API error')
  return res.json()
}
```

Características:
- Realiza petición POST al endpoint /diagnostico
- Envía el texto en formato JSON
- Maneja errores de red y del servidor
- Retorna la respuesta parseada como JSON

### Archivo: src/components/Chat.js

Componente principal que implementa la interfaz de chat.

#### Estructura del Componente

El componente crea una interfaz de chat completa con:
- Área de mensajes con scroll automático
- Campo de entrada de texto
- Botón de envío
- Estilos integrados en el componente

#### Elementos de la Interfaz

**1. Contenedor de Mensajes**

Muestra el historial de la conversación:
- Mensajes del usuario (fondo verde, alineados a la derecha)
- Mensajes del bot (fondo gris oscuro, alineados a la izquierda)
- Scroll automático al final cuando llega un mensaje nuevo

**2. Caja de Entrada**

Permite al usuario escribir mensajes:
- Input de texto con placeholder
- Botón de envío que se desactiva durante el procesamiento
- Soporte para Enter para enviar mensaje

#### Funciones Principales

**addMessage(container, text, cls)**

Añade un mensaje simple al chat:
- Crea un elemento párrafo con la clase especificada
- Añade el texto al elemento
- Lo inserta en el contenedor
- Hace scroll automático al final

**addTitle(container, title, text)**

Añade un mensaje con título resaltado:
- Útil para mostrar secciones de la respuesta del diagnóstico
- El título aparece en color dorado (amarillo)
- El texto aparece debajo del título

**mostrarResultado(container, data)**

Procesa y muestra la respuesta del diagnóstico:

1. **Muestra información básica:**
   - Texto recibido
   - Enfermedad detectada
   - Nivel de urgencia

2. **Filtra medicamentos:**
   - Solo muestra medicamentos con Match > 70%
   - Solo muestra medicamentos de la enfermedad detectada
   - Formatea cada medicamento con su porcentaje de match

3. **Maneja casos sin resultados:**
   - Muestra mensaje informativo si no hay medicamentos con alto match

**enviar()**

Función principal que maneja el envío de mensajes:

Proceso:
1. Valida que haya texto en el input
2. Muestra el mensaje del usuario en el chat
3. Desactiva el botón y muestra indicador de carga
4. Envía el mensaje al backend mediante chatApi
5. Muestra el resultado del diagnóstico
6. Maneja errores de conexión
7. Reactiva el botón y limpia el input
8. Enfoca el input para el siguiente mensaje

#### Estilos del Chat

El componente incluye estilos CSS integrados:

**Contenedor Principal (.chat):**
- Diseño vertical con flexbox
- Altura fija de 520px, ancho 420px
- Borde redondeado y sombras sutiles
- Tema oscuro con colores contrastantes

**Área de Mensajes (.messages):**
- Flexible, ocupa el espacio disponible
- Scroll vertical automático
- Padding para espaciado
- Fondo gris oscuro

**Mensajes de Usuario (.user):**
- Fondo verde para distinguir del bot
- Alineado a la derecha
- Bordes redondeados

**Mensajes del Bot (.bot):**
- Fondo gris más oscuro
- Alineado a la izquierda
- Bordes redondeados

**Títulos (.title):**
- Color dorado (amarillo) para resaltar
- Texto en negrita

**Caja de Entrada (.input-box):**
- Diseño horizontal con flexbox
- Input y botón con bordes redondeados
- Tema oscuro consistente
- Botón deshabilitado en gris durante carga

---

## Flujo de Datos

### Flujo Completo de un Diagnóstico

1. **Usuario escribe síntomas en el Frontend**
   - Ejemplo: "Tengo tos seca, me cuesta respirar y me duele el pecho"

2. **Frontend envía petición al Backend**
   - POST a http://localhost:8080/diagnostico
   - Body: { "texto": "Tengo tos seca..." }

3. **Backend analiza el texto**
   - Extrae palabras clave de síntomas: tos, respirar, pecho
   - Cuenta: 3 síntomas detectados
   - Detecta red flag de pecho: true
   - Detecta red flag de respiración: true

4. **Backend consulta HuggingFace**
   - Envía texto con token <mask>
   - Recibe probabilidades:
     - asma: 0.15
     - bronquitis: 0.68
     - enfisema: 0.10
     - otras: valores menores

5. **Backend construye vector de entrada**
   - Combina probabilidades de HuggingFace
   - Añade características del texto
   - Vector resultante: [0.15, 0.68, 0.10, ..., 3, 0, 1, 1, 0]

6. **Modelo Softmax clasifica**
   - Calcula scores: X·W + b
   - Aplica softmax para obtener probabilidades
   - Argmax determina la clase: 2 (bronquitis de alta urgencia)

7. **Backend mapea la clase a diagnóstico**
   - Clase 2 → enfermedad: bronquitis, urgencia: alta, cronica: si

8. **Backend consulta Prolog**
   - Query: medicamento_contraindicado(alta, bronquitis, cronica_si, pecho_si, resp_si, Med)
   - Prolog retorna todos los medicamentos contraindicados
   - Backend calcula porcentaje de match para cada uno

9. **Backend genera advertencias**
   - "Diagnostico: bronquitis"
   - "URGENCIA ALTA: Se recomienda atencion medica inmediata"
   - "RED FLAG: Dolor o presion en el pecho detectado"
   - "RED FLAG: Dificultad respiratoria severa detectada"

10. **Backend responde al Frontend**
    - JSON completo con toda la información procesada

11. **Frontend muestra resultados**
    - Texto recibido
    - Enfermedad detectada
    - Nivel de urgencia
    - Medicamentos contraindicados con Match > 70%

12. **RPA se ejecuta automáticamente**
    - Script PowerShell realiza acciones post-diagnóstico

### Diagrama de Flujo

```
Usuario → Frontend → Backend → Análisis Texto
                         ↓
                    HuggingFace API
                         ↓
                   Construcción Vector
                         ↓
                   Modelo Softmax
                         ↓
                   Mapeo Diagnóstico
                         ↓
                    Motor Prolog
                         ↓
                  Generación Respuesta
                         ↓
                      RPA Script
                         ↓
                     Frontend → Usuario
```

---

## Instalación y Configuración

### Requisitos Previos

**Backend:**
- Go 1.24.0 o superior
- Variables de entorno configuradas

**Frontend:**
- Node.js 16 o superior
- npm o yarn

### Configuración del Backend

**1. Instalar Go**

Descargar e instalar desde https://golang.org/

**2. Clonar el repositorio**

Navegar a la carpeta del proyecto.

**3. Configurar variables de entorno**

Crear archivo .env en la carpeta backend:
```
HF_TOKEN=tu_token_de_huggingface
```

Para obtener un token:
- Crear cuenta en https://huggingface.co/
- Ir a Settings → Access Tokens
- Crear un nuevo token con permisos de lectura

**4. Instalar dependencias**

```bash
cd backend
go mod download
```

**5. Entrenar el modelo inicial (opcional)**

Si no existe el archivo weights/softmax_model.json:

```bash
go run train_main.go
```

**6. Iniciar el servidor**

```bash
go run api.go
```

El servidor se iniciará en http://localhost:8080

### Configuración del Frontend

**1. Navegar a la carpeta del frontend**

```bash
cd Frontend/chat-frontend
```

**2. Instalar dependencias**

```bash
npm install
```

**3. Iniciar el servidor de desarrollo**

```bash
npm run dev
```

El frontend se abrirá en http://localhost:5173 (o el puerto que indique Vite)

### Verificación de la Instalación

**1. Verificar el backend**

Abrir navegador en http://localhost:8080

Debería mostrar información del servicio.

**2. Verificar el frontend**

Abrir navegador en http://localhost:5173

Debería mostrar la interfaz de chat.

**3. Probar el flujo completo**

En el frontend, escribir un mensaje de prueba:
```
Tengo tos seca y me cuesta respirar
```

El sistema debería responder con un diagnóstico completo.

### Solución de Problemas Comunes

**Error: HF_TOKEN no está configurado**
- Verificar que el archivo .env existe en la carpeta backend
- Verificar que la variable tiene un token válido

**Error: Modelo Softmax no encontrado**
- Ejecutar go run train_main.go para entrenar el modelo
- Verificar que existe weights/softmax_model.json

**Error: Prolog no carga**
- Verificar que existen los archivos en prolog/
- Verificar permisos de lectura de los archivos

**Error: Frontend no conecta con Backend**
- Verificar que el backend está corriendo en el puerto 8080
- Verificar que no hay firewall bloqueando la conexión
- Revisar la URL en chatApi.js

---

## Casos de Uso

### Caso 1: Diagnóstico de Asma con Alta Urgencia

**Entrada del Usuario:**
```
No puedo respirar bien, tengo silbidos en el pecho y tos desde hace años
```

**Procesamiento:**
1. Análisis de texto detecta: 3 síntomas, 1 crónica, red flag respiración
2. HuggingFace predice: asma 72%, bronquitis 18%
3. Softmax clasifica: Clase 1 (asma, urgencia mediana)
4. Prolog identifica medicamentos contraindicados para asma

**Respuesta Esperada:**
- Enfermedad: asma
- Urgencia: mediana
- Medicamentos contraindicados: betabloqueantes no selectivos
- Advertencia: Red flag respiratoria detectada

### Caso 2: Dolor de Pecho sin Enfermedad Respiratoria

**Entrada del Usuario:**
```
Siento presión en el pecho desde esta mañana
```

**Procesamiento:**
1. Análisis detecta: 1 síntoma, red flag pecho
2. HuggingFace: probabilidades bajas para todas las enfermedades
3. Softmax: Clase 0 (ninguna enfermedad específica)
4. Prolog considera el dolor de pecho

**Respuesta Esperada:**
- Enfermedad: ninguna
- Urgencia: baja o mediana
- Advertencia: Red flag de dolor de pecho detectado
- Recomendación: Evaluación médica para descartar causas cardíacas

### Caso 3: Condición Crónica Estable

**Entrada del Usuario:**
```
Tengo fibromialgia diagnosticada hace 5 años, hoy tengo dolor muscular
```

**Procesamiento:**
1. Análisis: 1 síntoma, 1 crónica
2. HuggingFace: fibromialgia 45%
3. Softmax: Clase 5 (fibromialgia, urgencia baja)
4. Prolog: medicamentos a evitar en fibromialgia

**Respuesta Esperada:**
- Enfermedad: fibromialgia
- Urgencia: baja
- Medicamentos contraindicados: corticoides sistémicos prolongados
- Sin advertencias críticas

### Caso 4: Crisis Respiratoria Severa

**Entrada del Usuario:**
```
Me ahogo, no puedo respirar, dolor intenso en el pecho, labios azules
```

**Procesamiento:**
1. Análisis: 4 síntomas, 2 red flags (pecho y respiración crítica)
2. HuggingFace: múltiples enfermedades respiratorias
3. Softmax: Clase 2 o 3 (urgencia alta)
4. Prolog: múltiples medicamentos contraindicados

**Respuesta Esperada:**
- Urgencia: ALTA
- Advertencias múltiples:
  - "URGENCIA ALTA: Se recomienda atención médica inmediata"
  - "RED FLAG: Dolor o presión en el pecho detectado"
  - "RED FLAG: Dificultad respiratoria severa detectada"
- Múltiples medicamentos contraindicados (sedantes, opioides, etc.)

---

## Consideraciones de Seguridad

### Advertencias Importantes

1. **No es un sistema de diagnóstico médico real:**
   - Este es un proyecto académico y de demostración
   - No debe usarse para tomar decisiones médicas reales
   - Siempre consultar con profesionales de la salud

2. **Protección de datos sensibles:**
   - El sistema no almacena información de pacientes
   - Cada diagnóstico es independiente y no persiste
   - No compartir datos médicos reales en el sistema

3. **Seguridad del token de API:**
   - Mantener el HF_TOKEN seguro y no compartirlo
   - No incluir el archivo .env en el control de versiones
   - Usar variables de entorno para información sensible

### Limitaciones Conocidas

1. **Precisión del modelo:**
   - El modelo es básico y entrenado con datos sintéticos
   - Las predicciones pueden no ser precisas en casos complejos

2. **Base de conocimiento limitada:**
   - Prolog contiene ejemplos didácticos, no conocimiento médico completo
   - Las contraindicaciones son simplificadas

3. **Análisis de texto básico:**
   - La extracción de características es mediante palabras clave
   - No comprende contexto complejo o negaciones sutiles

---

## Mantenimiento y Extensión

### Actualizar la Base de Conocimiento

Para añadir nuevas contraindicaciones:

1. Editar prolog/conocimiento.pl
2. Añadir nuevas reglas siguiendo el formato existente
3. Reiniciar el servidor backend

### Entrenar con Nuevos Datos

Para mejorar el modelo Softmax:

1. Preparar un CSV con el formato adecuado
2. Actualizar train.go con la ruta del nuevo dataset
3. Ejecutar go run train_main.go
4. Verificar la precisión del modelo entrenado

### Añadir Nuevas Enfermedades

1. Actualizar clasificacionMedica en api.go
2. Añadir keywords relevantes a los arrays de palabras clave
3. Añadir reglas en conocimiento.pl
4. Re-entrenar el modelo con la nueva clase

### Mejorar el Frontend

1. Modificar Chat.js para añadir nuevas funcionalidades
2. Actualizar estilos en style.css
3. Añadir nuevos componentes en src/components/

---

## Conclusión

Este sistema integra múltiples técnicas de inteligencia artificial para proporcionar un análisis médico asistido:

- **Procesamiento de Lenguaje Natural:** Comprensión del texto del paciente
- **Machine Learning:** Clasificación mediante regresión softmax
- **Sistemas Expertos:** Razonamiento lógico con Prolog
- **Integración de APIs:** Uso de modelos de lenguaje especializados

El proyecto demuestra cómo diferentes paradigmas de IA pueden trabajar juntos para resolver problemas complejos del mundo real, proporcionando una solución robusta y extensible.
