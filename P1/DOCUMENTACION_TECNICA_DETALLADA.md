# Documentación Técnica Detallada - Sistema UniMatch

## Tabla de Contenidos

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Backend - Análisis Profundo](#backend---análisis-profundo)
3. [Frontend - Análisis Profundo](#frontend---análisis-profundo)
4. [Algoritmos y Modelos](#algoritmos-y-modelos)
5. [Base de Datos y Persistencia](#base-de-datos-y-persistencia)
6. [API y Comunicación](#api-y-comunicación)
7. [Optimización y Rendimiento](#optimización-y-rendimiento)

---

## Arquitectura del Sistema

### Visión General

El sistema UniMatch implementa una arquitectura cliente-servidor desacoplada con múltiples componentes de inteligencia artificial trabajando en conjunto.

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Interfaz de Chat (JavaScript)              │  │
│  │  - Captura de entrada del usuario                    │  │
│  │  - Visualización de resultados                       │  │
│  │  - Gestión del estado de la UI                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP POST /diagnostico
                              │ JSON: {texto: "..."}
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND (Go)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    API REST (Fiber)                  │  │
│  │  - Validación de entrada                             │  │
│  │  - Orquestación del pipeline                         │  │
│  │  - Gestión de errores                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│              ┌───────────────┼───────────────┐             │
│              ▼               ▼               ▼             │
│  ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐    │
│  │  Análisis de    │ │   Cliente   │ │   Motor de   │    │
│  │     Texto       │ │  HuggingFace│ │    Prolog    │    │
│  │                 │ │             │ │              │    │
│  │ - Keywords      │ │ - NLP Model │ │ - Base de    │    │
│  │ - Red Flags     │ │ - Probs     │ │   Conocim.   │    │
│  │ - Contadores    │ │             │ │ - Inferencia │    │
│  └─────────────────┘ └─────────────┘ └──────────────┘    │
│              │               │               │             │
│              └───────────────┼───────────────┘             │
│                              ▼                              │
│                  ┌───────────────────────┐                 │
│                  │  Modelo Softmax (ML)  │                 │
│                  │                       │                 │
│                  │  - Feature Vector     │                 │
│                  │  - Clasificación      │                 │
│                  │  - Probabilidades     │                 │
│                  └───────────────────────┘                 │
│                              │                              │
│                              ▼                              │
│              ┌───────────────────────────────┐             │
│              │   Generador de Respuestas     │             │
│              │  - Mapeo de diagnósticos      │             │
│              │  - Evaluación de medicamentos │             │
│              │  - Generación de advertencias │             │
│              └───────────────────────────────┘             │
│                              │                              │
│                              ▼                              │
│                    ┌─────────────────┐                     │
│                    │   RPA Script    │                     │
│                    │  (PowerShell)   │                     │
│                    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ JSON Response
                              ▼
                          Usuario
```

### Componentes Principales

#### 1. Servidor API (Fiber Framework)

El servidor está construido con Fiber, un framework web de Go inspirado en Express.js pero optimizado para alto rendimiento.

**Características:**
- Manejo asíncrono de peticiones
- Middleware para CORS y manejo de errores
- Validación automática de JSON
- Respuestas tipadas y estructuradas

**Configuración:**
```go
app := fiber.New(fiber.Config{
    ErrorHandler: func(c *fiber.Ctx, err error) error {
        return c.Status(500).JSON(fiber.Map{
            "error": err.Error(),
        })
    },
})
```

#### 2. Procesador de Lenguaje Natural

**Componente Híbrido:**
- Análisis local basado en keywords (Go)
- Análisis contextual mediante HuggingFace (API externa)

**Flujo de procesamiento:**

1. **Preprocesamiento:**
   - Conversión a minúsculas
   - Tokenización implícita mediante búsqueda de subcadenas

2. **Extracción de características:**
   - Contadores numéricos (síntomas, condiciones crónicas)
   - Banderas booleanas (red flags)

3. **Análisis semántico:**
   - Modelo biomédico transformer
   - Generación de embeddings contextuales
   - Probabilidades por enfermedad

#### 3. Modelo de Machine Learning

**Implementación personalizada de Softmax Regression:**

Ventajas de implementación propia:
- Control total sobre el algoritmo
- Sin dependencias de frameworks pesados
- Optimizado para el caso de uso específico
- Fácil de modificar y extender

**Características técnicas:**
- Regularización L2 para evitar overfitting
- Descenso de gradiente por lotes
- Persistencia en formato JSON
- Soporte para múltiples clases

#### 4. Sistema Experto (Prolog)

**Motor de inferencia lógica:**
- Base de conocimiento declarativa
- Consultas mediante unificación
- Razonamiento hacia atrás
- Múltiples soluciones por consulta

**Ventajas del enfoque:**
- Separación clara entre conocimiento y lógica
- Fácil actualización de reglas
- Trazabilidad de decisiones
- Explicabilidad del razonamiento

---

## Backend - Análisis Profundo

### Gestión de Dependencias (go.mod)

El proyecto utiliza Go Modules para gestión de dependencias:

```go
module unmatch/backend

go 1.24.0

require (
    github.com/gofiber/fiber/v2 v2.52.8    // Framework web
    github.com/joho/godotenv v1.5.1        // Variables de entorno
    github.com/mndrix/golog v0.0.0         // Motor Prolog
    gonum.org/v1/gonum v0.16.0             // Matemáticas y ML
)
```

**Dependencias clave:**

1. **Fiber:** Framework web minimalista y rápido
   - Basado en Fasthttp
   - Bajo consumo de memoria
   - Alta concurrencia

2. **Gonum:** Suite matemática completa
   - Álgebra lineal (matrices densas y dispersas)
   - Estadística
   - Optimización numérica

3. **Golog:** Intérprete Prolog en Go
   - Compatibilidad con sintaxis Prolog estándar
   - Integración nativa con Go
   - Sin necesidad de procesos externos

### Estructura de Datos Avanzada

#### Vector de Características

El sistema construye un vector de 12 dimensiones:

```
Dimensión  │ Tipo    │ Descripción                      │ Rango
───────────┼─────────┼──────────────────────────────────┼──────────
0          │ float32 │ Probabilidad de asma             │ [0, 1]
1          │ float32 │ Probabilidad de bronquitis       │ [0, 1]
2          │ float32 │ Probabilidad de enfisema         │ [0, 1]
3          │ float32 │ Probabilidad de apnea            │ [0, 1]
4          │ float32 │ Probabilidad de fibromialgia     │ [0, 1]
5          │ float32 │ Probabilidad de migrañas         │ [0, 1]
6          │ float32 │ Probabilidad de reflujo          │ [0, 1]
7          │ int→f64 │ Número de síntomas detectados    │ [0, ∞)
8          │ int→f64 │ Número de condiciones crónicas   │ [0, ∞)
9          │ bool→f64│ Red flag: dolor de pecho         │ {0, 1}
10         │ bool→f64│ Red flag: dificultad respiratoria│ {0, 1}
11         │ bool→f64│ Tiene condiciones crónicas       │ {0, 1}
```

**Normalización:**
- Las probabilidades ya están normalizadas [0, 1]
- Los contadores no se normalizan (valores absolutos informativos)
- Los booleanos se convierten a 0 o 1

### Procesamiento de Texto en Profundidad

#### Algoritmo de Extracción de Características

```go
func analizarTexto(texto string) FeaturesTexto {
    textoLower := strings.ToLower(texto)
    var features FeaturesTexto

    // Fase 1: Detección de síntomas
    for _, sintoma := range sintomasKeywords {
        if strings.Contains(textoLower, sintoma) {
            features.n_sintomas++
        }
    }

    // Fase 2: Detección de condiciones crónicas
    for _, cronica := range cronicasKeywords {
        if strings.Contains(textoLower, cronica) {
            features.n_cronicas++
            features.tiene_cronicas = true
        }
    }

    // Fase 3: Detección de red flags críticas
    for _, keyword := range pechoKeywords {
        if strings.Contains(textoLower, keyword) {
            features.redflag_pecho = true
            break
        }
    }

    for _, keyword := range respiracionKeywords {
        if strings.Contains(textoLower, keyword) {
            features.redflag_respiracion = true
            break
        }
    }

    return features
}
```

**Complejidad computacional:**
- Conversión a minúsculas: O(n) donde n es la longitud del texto
- Búsqueda de keywords: O(k·n) donde k es el número de keywords
- Total: O(n) ya que k es constante

**Mejoras posibles:**
- Usar expresiones regulares para patrones complejos
- Implementar búsqueda de n-gramas para frases
- Añadir corrección ortográfica
- Implementar stemming para raíces de palabras

### Integración con HuggingFace

#### Modelo: bsc-bio-ehr-es

**Características del modelo:**
- Entrenado en texto médico en español
- Basado en arquitectura BERT
- Especializado en registros electrónicos de salud
- Comprende terminología médica técnica

#### Protocolo de Comunicación

```go
func llamarHuggingFace(texto string) (map[string]float64, error) {
    // 1. Preparación del payload
    textoConMask := texto + " padezco de <mask>."
    payload := map[string]string{
        "inputs": textoConMask,
    }
    
    // 2. Construcción de la petición HTTP
    req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
    req.Header.Set("Authorization", "Bearer "+token)
    req.Header.Set("Content-Type", "application/json")
    
    // 3. Envío y recepción
    resp, _ := http.DefaultClient.Do(req)
    
    // 4. Parseo de respuesta
    // Formato: [{"token_str": "asma", "score": 0.65}, ...]
    
    return probabilidades, nil
}
```

**Formato de respuesta de HuggingFace:**
```json
[
  {
    "score": 0.6532,
    "token": 1234,
    "token_str": "bronquitis",
    "sequence": "Tengo tos seca padezco de bronquitis."
  },
  {
    "score": 0.1823,
    "token": 5678,
    "token_str": "asma",
    "sequence": "Tengo tos seca padezco de asma."
  }
]
```

**Manejo de errores:**
- Timeout después de 30 segundos
- Reintentos automáticos en caso de error de red
- Fallback a probabilidades uniformes si falla

### Pipeline de Diagnóstico Detallado

#### Paso 1: Análisis de Texto Local

**Entrada:** Texto del paciente en lenguaje natural

**Proceso:**
1. Normalización del texto (lowercase)
2. Búsqueda de palabras clave mediante substring matching
3. Incremento de contadores por cada match
4. Activación de flags booleanas según keywords críticas

**Salida:** Objeto FeaturesTexto con métricas numéricas

**Ejemplo:**
```
Entrada: "Tengo tos desde hace años y me duele el pecho"

Procesamiento:
- "tos" → n_sintomas++
- "desde hace años" → n_cronicas++, tiene_cronicas=true
- "duele el pecho" → redflag_pecho=true

Salida: FeaturesTexto{
    n_sintomas: 1,
    n_cronicas: 1,
    redflag_pecho: true,
    redflag_respiracion: false,
    tiene_cronicas: true
}
```

#### Paso 2: Análisis Semántico (HuggingFace)

**Entrada:** Texto original con token de predicción

**Proceso:**
1. Construcción del prompt: texto + " padezco de <mask>."
2. Envío a la API de HuggingFace
3. El modelo predice las palabras más probables para <mask>
4. Extracción de scores para enfermedades conocidas

**Salida:** Mapa de enfermedad → probabilidad

**Ejemplo:**
```
Entrada: "Tengo tos desde hace años y me duele el pecho padezco de <mask>."

API Response:
- bronquitis: 0.68
- asma: 0.15
- neumonía: 0.08
- gripe: 0.05
- otros: 0.04

Salida: map[string]float64{
    "bronquitis": 0.68,
    "asma": 0.15,
    ...
}
```

#### Paso 3: Construcción del Vector de Entrada

**Proceso de fusión:**
```go
entrada := VectorEntrada{
    // Probabilidades del modelo NLP
    a_asma:         float32(probabilidadesHF["asma"]),
    a_bronquitis:   float32(probabilidadesHF["bronquitis"]),
    // ... más enfermedades
    
    // Características del análisis de texto
    n_sintomas:          featuresTexto.n_sintomas,
    n_cronicas:          featuresTexto.n_cronicas,
    redflag_pecho:       featuresTexto.redflag_pecho,
    redflag_respiracion: featuresTexto.redflag_respiracion,
    tiene_cronicas:      featuresTexto.tiene_cronicas,
}
```

**Vector resultante:**
```
[0.15, 0.68, 0.10, 0.02, 0.01, 0.01, 0.03, 2, 1, 1, 0, 1]
 │     │     │     │     │     │     │     │  │  │  │  │
 │     │     │     │     │     │     │     │  │  │  │  └─ tiene_cronicas
 │     │     │     │     │     │     │     │  │  │  └──── redflag_respiracion
 │     │     │     │     │     │     │     │  │  └─────── redflag_pecho
 │     │     │     │     │     │     │     │  └────────── n_cronicas
 │     │     │     │     │     │     │     └───────────── n_sintomas
 │     │     │     │     │     │     └─────────────────── a_reflujo
 │     │     │     │     │     └───────────────────────── a_migranas
 │     │     │     │     └─────────────────────────────── a_fibromialgia
 │     │     │     └───────────────────────────────────── a_apnea
 │     │     └─────────────────────────────────────────── a_enfisema
 │     └───────────────────────────────────────────────── a_bronquitis
 └─────────────────────────────────────────────────────── a_asma
```

#### Paso 4: Clasificación con Softmax

**Entrada:** Vector de 12 dimensiones

**Proceso matemático:**

1. **Cálculo de scores:**
   ```
   scores = X·W + b
   
   Donde:
   - X: vector de entrada (1×12)
   - W: matriz de pesos (12×8)
   - b: vector de sesgos (8)
   - scores: vector de scores (1×8)
   ```

2. **Aplicación de softmax:**
   ```
   P(y=k|X) = exp(scores[k]) / Σ exp(scores[i])
   
   Resultado: vector de probabilidades que suma 1
   ```

3. **Clasificación:**
   ```
   clase_predicha = argmax(probabilidades)
   ```

**Salida:**
- Clase predicha (0-7)
- Vector de probabilidades para todas las clases

**Ejemplo:**
```
scores = [1.2, 3.8, 2.1, 0.5, -0.3, 0.8, 1.1, 0.2]

Después de softmax:
probabilidades = [0.05, 0.68, 0.13, 0.02, 0.01, 0.03, 0.05, 0.03]

argmax = 1 → Clase 1 (asma, mediana urgencia, crónica)
```

#### Paso 5: Mapeo a Diagnóstico Médico

**Tabla de mapeo:**
```go
clasificacionMedica[1] = {
    Urgencia:   "mediana",
    Enfermedad: "asma",
    Cronica:    "cronica_si",
}
```

**Transformación adicional:**
```go
pecho := "pecho_no"
if entrada.redflag_pecho {
    pecho = "pecho_si"
}

respiracion := "resp_no"
if entrada.redflag_respiracion {
    respiracion = "resp_si"
}
```

**Salida:** Contexto médico completo para consulta Prolog

#### Paso 6: Consulta al Sistema Experto

**Construcción de la consulta Prolog:**
```prolog
medicamento_contraindicado(mediana, asma, cronica_si, pecho_si, resp_no, Med)
```

**Proceso de unificación:**
1. Prolog busca hechos que unifiquen con la consulta
2. Instancia la variable Med con cada solución
3. Retorna todas las soluciones posibles

**Ejemplo de soluciones:**
```prolog
Med = betabloqueantes_no_selectivos
Med = aines_altas_dosis
Med = sedantes_fuertes
```

**Cálculo de match:**

Para cada solución, se calcula un porcentaje de coincidencia:

```go
matchCount := 0

if urg_prolog == urg_detectada { matchCount++ }
if enf_prolog == enf_detectada { matchCount++ }
if cron_prolog == cron_detectada { matchCount++ }
if pecho_prolog == pecho_detectado { matchCount++ }
if resp_prolog == resp_detectada { matchCount++ }

// Bonificaciones por condiciones críticas
if urgencia == "alta" && respiracion == "resp_si" {
    matchCount++
}

if (enfermedad in enfermedades_pulmonares) && 
   cronica == "cronica_si" && 
   respiracion == "resp_si" {
    matchCount++
}

matchPercent := (matchCount / 7.0) * 100.0
```

**Máximo match posible:** 7 puntos = 100%

#### Paso 7: Generación de Advertencias

**Sistema de prioridad de advertencias:**

1. **Nivel crítico (urgencia alta):**
   ```
   "URGENCIA ALTA: Se recomienda atención médica inmediata"
   ```

2. **Red flags detectadas:**
   ```
   "RED FLAG: Dolor o presión en el pecho detectado"
   "RED FLAG: Dificultad respiratoria severa detectada"
   ```

3. **Información diagnóstica:**
   ```
   "Diagnóstico: [enfermedad]"
   ```

4. **Contraindicaciones:**
   ```
   "IMPORTANTE: X medicamentos están contraindicados para esta enfermedad"
   ```

**Lógica de generación:**
```go
var advertencias []string

if diagnostico.Enfermedad != "ninguna" {
    advertencias = append(advertencias,
        fmt.Sprintf("Diagnostico: %s", diagnostico.Enfermedad))
}

if diagnostico.Urgencia == "alta" {
    advertencias = append(advertencias,
        "URGENCIA ALTA: Se recomienda atencion medica inmediata")
}

if featuresTexto.redflag_pecho {
    advertencias = append(advertencias,
        "RED FLAG: Dolor o presion en el pecho detectado")
}

// ... más advertencias
```

#### Paso 8: Ejecución RPA

**Script PowerShell automático:**

Después de completar el diagnóstico, el sistema ejecuta un script RPA que puede:
- Registrar el diagnóstico en un archivo
- Enviar notificaciones
- Actualizar bases de datos externas
- Generar reportes

```go
cmd := exec.Command(
    "powershell",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "rpa.ps1",
)

cmd.Dir = exePath
out, err := cmd.CombinedOutput()
```

---

## Algoritmos y Modelos

### Regresión Softmax - Implementación Detallada

#### Teoría Matemática

**Función de activación softmax:**

Para un vector de scores z = [z₁, z₂, ..., zₖ]:

```
σ(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

**Propiedades:**
- Salida en rango (0, 1)
- Suma de todas las salidas = 1
- Interpretable como distribución de probabilidad

**Función de pérdida cross-entropy:**

Para una muestra con etiqueta verdadera y y predicción ŷ:

```
L = -Σᵢ yᵢ·log(ŷᵢ)

En formato one-hot, solo el término correcto contribuye:
L = -log(ŷₖ) donde k es la clase verdadera
```

**Regularización L2:**

```
L_total = L_cross_entropy + λ/2·||W||²

Donde:
- λ: fuerza de regularización
- ||W||²: suma de cuadrados de todos los pesos
```

#### Implementación del Forward Pass

```go
func (m *SoftmaxRegression) forward(X *mat.Dense) (*mat.Dense, *mat.Dense) {
    nSamples, _ := X.Dims()
    _, nClasses := m.W.Dims()

    // Paso 1: Cálculo de scores
    // scores[i,k] = Σⱼ X[i,j]·W[j,k]
    scores := mat.NewDense(nSamples, nClasses, nil)
    scores.Mul(X, m.W)

    // Paso 2: Añadir sesgo
    // scores[i,k] += b[k]
    for i := 0; i < nSamples; i++ {
        row := scores.RawRowView(i)
        for k := 0; k < nClasses; k++ {
            row[k] += m.B.AtVec(k)
        }
    }

    // Paso 3: Aplicar softmax
    probs := softmaxRows(scores)
    
    return scores, probs
}
```

#### Estabilidad Numérica en Softmax

**Problema:** exp(x) puede causar overflow para x grande

**Solución:** Trick de estabilidad numérica

```go
func softmaxRows(scores *mat.Dense) *mat.Dense {
    r, c := scores.Dims()
    out := mat.NewDense(r, c, nil)

    for i := 0; i < r; i++ {
        row := scores.RawRowView(i)
        outRow := out.RawRowView(i)

        // Encontrar máximo para estabilidad
        maxVal := row[0]
        for k := 1; k < c; k++ {
            if row[k] > maxVal {
                maxVal = row[k]
            }
        }

        // Calcular exp(x - max) en lugar de exp(x)
        sumExp := 0.0
        for k := 0; k < c; k++ {
            e := math.Exp(row[k] - maxVal)
            outRow[k] = e
            sumExp += e
        }

        // Normalizar
        for k := 0; k < c; k++ {
            outRow[k] /= sumExp
        }
    }
    return out
}
```

**Por qué funciona:**

```
σ(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
      = exp(zᵢ - max) / Σⱼ exp(zⱼ - max)

Esta expresión es matemáticamente equivalente pero más estable.
```

#### Algoritmo de Entrenamiento

**Descenso de gradiente por lotes:**

```go
func (m *SoftmaxRegression) Fit(X *mat.Dense, y []int) {
    // Inicialización
    nSamples, nFeatures := X.Dims()
    nClasses := max(y) + 1
    
    // Inicializar W con valores aleatorios pequeños
    m.W = initializeWeights(nFeatures, nClasses)
    m.B = zeros(nClasses)
    
    // One-hot encoding de etiquetas
    Y := oneHotDense(y, nSamples, nClasses)
    
    // Bucle de entrenamiento
    for iter := 0; iter < m.NIter; iter++ {
        // 1. Forward pass
        _, probs := m.forward(X)
        
        // 2. Calcular pérdida
        loss := computeLoss(Y, probs, m.W, m.RegLambda)
        m.LossHistory = append(m.LossHistory, loss)
        
        // 3. Calcular gradientes
        dScores := probs - Y  // Gradiente de scores
        dScores /= nSamples
        
        dW := X.T() · dScores + λ·W  // Gradiente de W
        db := sum_rows(dScores)       // Gradiente de b
        
        // 4. Actualizar parámetros
        m.W = m.W - lr·dW
        m.B = m.B - lr·db
    }
}
```

**Derivación del gradiente:**

Para la pérdida cross-entropy con softmax:

```
∂L/∂zᵢ = ŷᵢ - yᵢ

Este resultado notable simplifica enormemente el cálculo.
```

#### Regularización L2

**Propósito:**
- Evitar overfitting
- Penalizar pesos grandes
- Mejorar generalización

**Efecto en la pérdida:**
```
L_reg = L + λ/2·Σᵢⱼ Wᵢⱼ²
```

**Efecto en el gradiente:**
```
∇W_reg = ∇W + λ·W
```

**Elección de λ:**
- λ = 0: Sin regularización
- λ pequeño (0.001): Regularización suave
- λ grande (1.0): Regularización fuerte

### Motor Prolog - Análisis del Sistema Experto

#### Sintaxis de Prolog

**Hechos:**
```prolog
medicamento_contraindicado(alta, asma, cronica_si, pecho_si, resp_si, sedantes_fuertes).
```

**Estructura:**
- Predicado: medicamento_contraindicado
- Aridad: 6 (seis argumentos)
- Argumentos: constantes (átomos)

**Consultas:**
```prolog
?- medicamento_contraindicado(alta, asma, cronica_si, pecho_si, resp_si, Med).
```

**Variables:**
- Comienzan con mayúscula: Med, Urg, Enf
- Unificación: proceso de asignar valores a variables

#### Algoritmo de Unificación

**Proceso:**

1. **Parseo de la consulta**
2. **Búsqueda en la base de conocimiento**
3. **Intento de unificación con cada hecho**
4. **Retorno de soluciones**

**Ejemplo de unificación:**

Consulta:
```prolog
?- medicamento_contraindicado(alta, Enf, cronica_si, pecho_si, resp_si, Med).
```

Hecho 1:
```prolog
medicamento_contraindicado(alta, asma, cronica_si, pecho_si, resp_si, sedantes_fuertes).
```

**Proceso de unificación:**
```
alta == alta ✓
Enf unifica con asma → Enf = asma
cronica_si == cronica_si ✓
pecho_si == pecho_si ✓
resp_si == resp_si ✓
Med unifica con sedantes_fuertes → Med = sedantes_fuertes
```

**Resultado:** Solución encontrada con Enf=asma, Med=sedantes_fuertes

#### Integración Go-Prolog

**Carga de la base de conocimiento:**
```go
programa := cargarProlog("./prolog/conocimiento.pl")
maquinaProlog := golog.NewMachine().Consult(programa)
```

**Ejecución de consultas:**
```go
query := "medicamento_contraindicado(Urg, Enf, Cron, Pecho, Resp, Med)."
solutions := maquinaProlog.ProveAll(query)

for _, sol := range solutions {
    urg := sol.ByName_("Urg").String()
    enf := sol.ByName_("Enf").String()
    med := sol.ByName_("Med").String()
    // Procesar solución
}
```

---

## Frontend - Análisis Profundo

### Arquitectura del Cliente

#### Patrón de Diseño: Component-Based

El frontend utiliza un enfoque modular sin framework:

```javascript
// Componente como función pura
export function Chat(container) {
  // 1. Renderizar HTML
  container.innerHTML = template
  
  // 2. Obtener referencias del DOM
  const messages = container.querySelector('#messages')
  const input = container.querySelector('#text')
  const button = container.querySelector('#send')
  
  // 3. Definir handlers
  const enviar = async () => { /* ... */ }
  
  // 4. Attachear event listeners
  button.onclick = enviar
  input.onkeydown = handleKeyPress
}
```

**Ventajas:**
- Sin overhead de frameworks
- Control total del DOM
- Fácil de entender y mantener
- Carga rápida

### Gestión del Estado

**Estado implícito en el DOM:**
- El historial de mensajes está en el DOM
- El texto de entrada está en el input
- El estado de carga está en el botón

**No hay estado global explícito:**
- Cada interacción es independiente
- No se mantiene historial en memoria
- Simplifica el flujo de datos

### Comunicación HTTP

#### Cliente Fetch API

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

**Características:**
- Basado en Promesas (async/await)
- Manejo de errores con try/catch
- Parsing automático de JSON
- Headers correctos para REST

#### Manejo de Respuestas

**Flujo de datos:**

1. **Recepción de JSON:**
```javascript
{
  "enfermedad_detectada": "bronquitis",
  "nivel_urgencia": "alta",
  "medicamentos_evaluados": [...]
}
```

2. **Filtrado de datos:**
```javascript
const medicamentosFiltrados = data.medicamentos_evaluados
  .filter(m => m.Match > 70 && m.Enfermedad === enfermedad)
  .map(m => `• ${m.Medicamento} (${m.Match.toFixed(1)}%)`)
```

3. **Renderizado:**
```javascript
if (medicamentosFiltrados.length > 0) {
  addTitle(container, 'Medicamentos no recomendados',
           medicamentosFiltrados.join('\n'))
} else {
  addTitle(container, 'Medicamentos no recomendados',
           'No se encontraron medicamentos con Match > 70%')
}
```

### Sistema de Eventos

#### Event Delegation

```javascript
button.onclick = enviar

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault()
    enviar()
  }
})
```

**Ventajas:**
- Listeners únicos, no múltiples
- Fácil de rastrear el flujo de eventos
- Prevención de comportamiento por defecto

### Renderizado Dinámico

#### Manipulación del DOM

**Creación de elementos:**
```javascript
function addMessage(container, text, cls) {
  const p = document.createElement('p')
  p.className = cls
  p.textContent = text
  container.appendChild(p)
  container.scrollTop = container.scrollHeight
}
```

**Scroll automático:**
```javascript
container.scrollTop = container.scrollHeight
```

Esto asegura que siempre se vea el mensaje más reciente.

#### Template Strings

**HTML como string:**
```javascript
container.innerHTML = `
  <style>...</style>
  <div class="chat">
    <div class="messages" id="messages"></div>
    <div class="input-box">
      <input id="text" placeholder="Escribe un mensaje" />
      <button id="send">Enviar</button>
    </div>
  </div>
`
```

**Ventajas:**
- Sintaxis limpia y legible
- Interpolación de variables con ${}
- Soporte para multi-línea

---

## Optimización y Rendimiento

### Backend

#### Manejo de Concurrencia

**Go routines implícitas:**
- Fiber maneja cada petición en una goroutine
- Concurrencia automática sin código explícito
- Pool de workers para reutilización

**Caché del modelo:**
```go
var softmaxModel *algorithms.SoftmaxRegression

// Primera petición: cargar modelo
if softmaxModel == nil {
    model, err := algorithms.LoadSoftmaxRegression(path)
    softmaxModel = model
}

// Peticiones subsecuentes: reutilizar modelo cargado
prediccion := softmaxModel.Predict(X)
```

**Ventaja:** Evita cargar el modelo en cada petición

#### Optimización de Matrices

**Gonum usa operaciones optimizadas:**
- BLAS (Basic Linear Algebra Subprograms)
- Operaciones vectorizadas
- Aprovecha múltiples núcleos cuando es posible

**Acceso directo a memoria:**
```go
row := matrix.RawRowView(i)
// Acceso directo al slice subyacente, sin copias
```

### Frontend

#### Lazy Loading

**Vite optimiza automáticamente:**
- Code splitting
- Tree shaking
- Minificación
- Compresión

**Carga diferida de módulos:**
```javascript
import('./components/Chat.js').then(module => {
  module.Chat(container)
})
```

#### Optimización de Renderizado

**Evitar reflows innecesarios:**
- Crear elemento completo antes de insertarlo
- Insertar una vez en el DOM
- Scroll automático después de inserción

```javascript
const p = document.createElement('p')
p.className = cls
p.textContent = text
// Solo una operación de DOM
container.appendChild(p)
```

### Consideraciones de Escalabilidad

#### Backend

**Para mayor escala, considerar:**

1. **Caché de respuestas:**
   - Redis para resultados frecuentes
   - TTL configurable
   - Invalidación selectiva

2. **Load balancing:**
   - Múltiples instancias del servidor
   - Nginx como reverse proxy
   - Health checks

3. **Base de datos:**
   - PostgreSQL para persistencia
   - Índices en campos frecuentemente consultados
   - Conexión pooling

4. **Procesamiento asíncrono:**
   - Cola de mensajes (RabbitMQ)
   - Workers separados para ML
   - Respuestas mediante webhooks

#### Frontend

**Para mayor escala, considerar:**

1. **Framework moderno:**
   - React/Vue/Svelte para aplicaciones complejas
   - Estado global (Redux, Vuex)
   - Routing para múltiples vistas

2. **PWA (Progressive Web App):**
   - Service workers para offline
   - Caché de assets
   - Notificaciones push

3. **Optimización de imágenes:**
   - Formatos modernos (WebP, AVIF)
   - Lazy loading de imágenes
   - CDN para assets estáticos

---

## Conclusión Técnica

Este sistema demuestra una integración efectiva de múltiples paradigmas y tecnologías:

1. **Paradigma imperativo:** Go para la lógica de negocio
2. **Paradigma funcional:** JavaScript moderno con funciones puras
3. **Paradigma lógico:** Prolog para razonamiento declarativo
4. **Machine Learning:** Implementación personalizada de softmax
5. **Deep Learning:** Modelo transformer para NLP

La arquitectura es modular, escalable y mantenible, con separación clara de responsabilidades entre componentes.
