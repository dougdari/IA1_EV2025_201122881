package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/joho/godotenv"
	"github.com/mndrix/golog"
	"gonum.org/v1/gonum/mat"

	"unmatch/backend/algorithms"
)

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

// MedicamentoEvaluado representa un medicamento con su evaluación de contraindicación

type MedicamentoRecomendado struct {
	Urgencia    string
	Enfermedad  string
	Cronica     string
	Pecho       string
	Respiracion string
	Medicamento string
	Match       float64
}

// DiagnosticoRequest es la solicitud del cliente con el texto médico
type DiagnosticoRequest struct {
	Texto string `json:"texto"`
}

// DiagnosticoResponse es la respuesta con medicamentos evaluados
type DiagnosticoResponse struct {
	EnfermedadDetectada       string                   `json:"enfermedad_detectada"`
	NivelUrgencia             string                   `json:"nivel_urgencia"`
	ProbabilidadesHuggingFace map[string]float64       `json:"probabilidades_huggingface"`
	ClaseSoftmax              int                      `json:"clase_softmax"`
	MedicamentosEvaluados     []MedicamentoRecomendado `json:"medicamentos_evaluados"`
	TotalContraindicados      int                      `json:"total_contraindicados"`
	Advertencias              []string                 `json:"advertencias"`
	TextoRecibido             string                   `json:"texto_recibido"`
}

// VectorEntrada contiene todas las características extraídas para el modelo
type VectorEntrada struct {
	// Probabilidades del modelo HuggingFace (0.0 - 1.0)
	a_asma         float32
	a_bronquitis   float32
	a_enfisema     float32
	a_apnea        float32
	a_fibromialgia float32
	a_migranas     float32
	a_reflujo      float32

	// Features extraídas del análisis de texto
	n_sintomas          int
	n_cronicas          int
	redflag_pecho       bool
	redflag_respiracion bool
	tiene_cronicas      bool
}

// FeaturesTexto contiene las características extraídas del análisis de texto
type FeaturesTexto struct {
	n_sintomas          int
	n_cronicas          int
	redflag_pecho       bool
	redflag_respiracion bool
	tiene_cronicas      bool
}

// ============================================================================
// MAPEOS
// ============================================================================

// Mapeo de clases Softmax a enfermedades y niveles de urgencia
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

// Keywords para detectar síntomas en el texto
var sintomasKeywords = []string{
	"pecho", "tos", "flema", "silbido", "falta de aire",
	"ahogo", "dificultad para respirar", "opresion", "dolor al respirar",
	"cansancio", "fatiga", "sibilancias", "esputo", "mucosidad",
}

// Keywords para detectar enfermedades crónicas en el texto
var cronicasKeywords = []string{
	"asma", "epoc", "bronquitis cronica", "fibrosis pulmonar",
	"enfisema", "apnea", "cronico", "cronica", "años de", "desde hace",
}

// Keywords que indican problemas en el pecho (red flag)
var pechoKeywords = []string{
	"dolor de pecho", "opresion en el pecho", "presion en el pecho",
	"pecho apretado", "dolor toracico", "dolor intenso en el pecho",
}

// Keywords que indican problemas respiratorios graves (red flag)
var respiracionKeywords = []string{
	"no puedo respirar", "falta de aire severa", "ahogo",
	"dificultad extrema", "labios azules", "cianosis", "me ahogo",
}

// ============================================================================
// VARIABLES GLOBALES
// ============================================================================

var softmaxModel *algorithms.SoftmaxRegression
var maquinaProlog golog.Machine

const softmaxModelPath = algorithms.DefaultSoftmaxModelPath

// ============================================================================
// FUNCIONES DE UTILIDAD
// ============================================================================

// cargarProlog carga el archivo de conocimiento Prolog
func cargarProlog(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Sprintf("Error al cargar Prolog: %v", err))
	}
	return string(data)
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// slice2DToDense convierte un slice 2D de Go a una matriz Gonum Dense
func slice2DToDense(x [][]float64) (*mat.Dense, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("la matriz X no puede estar vacía")
	}
	nSamples := len(x)
	nFeatures := len(x[0])
	data := make([]float64, 0, nSamples*nFeatures)

	for i := 0; i < nSamples; i++ {
		if len(x[i]) != nFeatures {
			return nil, fmt.Errorf("todas las filas de X deben tener el mismo número de columnas")
		}
		data = append(data, x[i]...)
	}
	return mat.NewDense(nSamples, nFeatures, data), nil
}

// denseTo2D convierte una matriz Gonum Dense a un slice 2D de Go
func denseTo2D(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := m.RawRowView(i)
		dst := make([]float64, c)
		copy(dst, row)
		out[i] = dst
	}
	return out
}

// ============================================================================
// ANÁLISIS DE TEXTO
// ============================================================================

// analizarTexto extrae características del texto del paciente
func analizarTexto(texto string) FeaturesTexto {
	textoLower := strings.ToLower(texto)

	var features FeaturesTexto

	// Contar síntomas mencionados
	for _, sintoma := range sintomasKeywords {
		if strings.Contains(textoLower, sintoma) {
			features.n_sintomas++
		}
	}

	// Contar enfermedades crónicas mencionadas
	for _, cronica := range cronicasKeywords {
		if strings.Contains(textoLower, cronica) {
			features.n_cronicas++
			features.tiene_cronicas = true
		}
	}

	// Detectar red flags de pecho
	for _, keyword := range pechoKeywords {
		if strings.Contains(textoLower, keyword) {
			features.redflag_pecho = true
			break
		}
	}

	// Detectar red flags de respiración
	for _, keyword := range respiracionKeywords {
		if strings.Contains(textoLower, keyword) {
			features.redflag_respiracion = true
			break
		}
	}

	return features
}

// ============================================================================
// INTEGRACIÓN CON HUGGINGFACE
// ============================================================================

// llamarHuggingFace envía el texto al modelo de HuggingFace para análisis NLP
func llamarHuggingFace(texto string) (map[string]float64, error) {
	fmt.Println("→ Llamando a HuggingFace API...")

	token := os.Getenv("HF_TOKEN")
	if token == "" {
		return nil, fmt.Errorf("la variable de entorno HF_TOKEN no está configurada")
	}

	url := "https://router.huggingface.co/hf-inference/models/PlanTL-GOB-ES/bsc-bio-ehr-es"

	// Agregar <mask> al texto para el modelo de fill-mask
	textoConMask := texto + " padezco de <mask>."

	payload, err := json.Marshal(map[string]string{
		"inputs": textoConMask,
	})
	if err != nil {
		return nil, fmt.Errorf("error al crear payload: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("error al crear request: %v", err)
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error al hacer request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error al leer respuesta: %v", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("error de HuggingFace: %s - %s", resp.Status, string(body))
	}

	// Parsear la respuesta
	var resultado interface{}
	if err := json.Unmarshal(body, &resultado); err != nil {
		return nil, fmt.Errorf("error al parsear JSON: %v", err)
	}

	// Convertir a mapa de probabilidades
	probabilidades := make(map[string]float64)

	// La respuesta viene como: [{"token_str": "asma", "score": 0.85}, ...]
	if resultArray, ok := resultado.([]interface{}); ok && len(resultArray) > 0 {
		for _, item := range resultArray {
			if itemMap, ok := item.(map[string]interface{}); ok {
				if token, ok := itemMap["token_str"].(string); ok {
					if score, ok := itemMap["score"].(float64); ok {
						probabilidades[token] = score
					}
				}
			}
		}
	}

	fmt.Printf("✓ HuggingFace: %d enfermedades detectadas\n", len(probabilidades))
	return probabilidades, nil
}

// ============================================================================
// EVALUACIÓN DE MEDICAMENTOS CON PROLOG
// ============================================================================

// obtenerMedicamentosContraindicados consulta Prolog DIRECTAMENTE
// para obtener los medicamentos contraindicados según el diagnóstico

func obtenerMedicamentosContraindicados(
	m golog.Machine,
	urgencia, enfermedad, cronica, pecho, respiracion string,
) []MedicamentoRecomendado {

	// Usamos la regla de medicamentos contraindicados
	query := "medicamento_contraindicado(Urg, Enf, Cron, Pecho, Resp, Med)."
	solutions := m.ProveAll(query)

	results := []MedicamentoRecomendado{}

	for _, sol := range solutions {
		urg := sol.ByName_("Urg").String()
		enf := sol.ByName_("Enf").String()
		cron := sol.ByName_("Cron").String()
		pech := sol.ByName_("Pecho").String()
		resp := sol.ByName_("Resp").String()
		med := sol.ByName_("Med").String()

		matchCount := 0

		// 1: urgencia

		fmt.Println("Evaluando medicamento:", med)
		fmt.Println(urg, enf, cron, pech, resp)
		fmt.Println(urgencia, enfermedad, cronica, pecho, respiracion)
		fmt.Println(urg, urgencia)

		if urg == urgencia {
			matchCount++
		}
		// 2: enfermedad

		fmt.Println(enf, enfermedad)
		if enf == enfermedad {
			matchCount++
		}
		// 3: cronicidad

		fmt.Println(cron, cronica)
		if cron == cronica {
			matchCount++
		}
		// 4: pecho

		fmt.Println(pech, pecho)
		if pech == pecho {
			matchCount++
		}
		// 5: respiración

		fmt.Println(resp, respiracion)
		if resp == respiracion {
			matchCount++
		}

		// 6: “peso extra” por combinación peligrosa (ejemplos)
		// alta urgencia + problema respiratorio

		fmt.Println(urgencia, respiracion)
		if urgencia == "alta" && respiracion == "resp_si" {
			matchCount++
		}
		// enfermedad pulmonar crónica + respiración comprometida

		if (enfermedad == "asma" || enfermedad == "bronquitis" || enfermedad == "enfisema") &&
			cronica == "cronica_si" && respiracion == "resp_si" {
			matchCount++
		}

		// Tenemos 5 condiciones base + 2 extras posibles = divisor 7.0
		matchPercent := float64(matchCount) / 7.0 * 100.0

		results = append(results, MedicamentoRecomendado{
			Urgencia:    urg,
			Enfermedad:  enf,
			Cronica:     cron,
			Pecho:       pech,
			Respiracion: resp,
			Medicamento: med,
			Match:       matchPercent,
		})
	}

	return results
}

// ============================================================================
// PIPELINE COMPLETO DE DIAGNÓSTICO
// ============================================================================

// procesarDiagnostico ejecuta todo el pipeline
func procesarDiagnostico(req DiagnosticoRequest) (*DiagnosticoResponse, error) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("INICIANDO DIAGNÓSTICO MÉDICO")
	fmt.Println(strings.Repeat("=", 60))

	// ========== PASO 1: ANÁLISIS DE TEXTO ==========
	fmt.Println("\n[PASO 1] Análisis de texto del paciente")
	featuresTexto := analizarTexto(req.Texto)
	fmt.Printf("  • Síntomas detectados: %d\n", featuresTexto.n_sintomas)
	fmt.Printf("  • Enfermedades crónicas: %d\n", featuresTexto.n_cronicas)
	fmt.Printf("  • Red flag pecho: %v\n", featuresTexto.redflag_pecho)
	fmt.Printf("  • Red flag respiración: %v\n", featuresTexto.redflag_respiracion)

	// ========== PASO 2: HUGGINGFACE NLP ==========
	fmt.Println("\n[PASO 2] Análisis con HuggingFace (NLP)")
	probabilidadesHF, err := llamarHuggingFace(req.Texto)
	if err != nil {
		return nil, fmt.Errorf("error en HuggingFace: %v", err)
	}

	// Construir vector de entrada con probabilidades de HF
	var entrada VectorEntrada
	entrada.a_asma = float32(probabilidadesHF["asma"])
	entrada.a_bronquitis = float32(probabilidadesHF["bronquitis"])
	entrada.a_enfisema = float32(probabilidadesHF["enfisema"])
	entrada.a_apnea = float32(probabilidadesHF["apnea"])
	entrada.a_fibromialgia = float32(probabilidadesHF["fibromialgia"])
	entrada.a_migranas = float32(probabilidadesHF["migrañas"])
	entrada.a_reflujo = float32(probabilidadesHF["reflujo"])
	entrada.n_sintomas = featuresTexto.n_sintomas
	entrada.n_cronicas = featuresTexto.n_cronicas
	entrada.redflag_pecho = featuresTexto.redflag_pecho
	entrada.redflag_respiracion = featuresTexto.redflag_respiracion
	entrada.tiene_cronicas = featuresTexto.tiene_cronicas

	// ========== PASO 3: CLASIFICACIÓN CON SOFTMAX ==========
	fmt.Println("\n[PASO 3] Clasificación con modelo Softmax")

	// Verificar que el modelo esté cargado
	if softmaxModel == nil {
		if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
			softmaxModel = model
			fmt.Println("  ⚠ Modelo cargado desde disco")
		} else {
			return nil, fmt.Errorf("modelo Softmax no disponible. Entrénelo primero")
		}
	}

	// Crear matriz de entrada para Softmax (1 muestra x 9 features)
	Xdata := []float64{

		float64(entrada.a_asma),
		float64(entrada.a_bronquitis),
		float64(entrada.a_enfisema),
		float64(entrada.a_apnea),
		float64(entrada.a_fibromialgia),
		float64(entrada.a_migranas),
		float64(entrada.a_reflujo),
		float64(entrada.n_sintomas),
		float64(entrada.n_cronicas),
		boolToFloat(entrada.redflag_pecho),
		boolToFloat(entrada.redflag_respiracion),
		boolToFloat(entrada.tiene_cronicas),
	}
	Xmat := mat.NewDense(1, len(Xdata), Xdata)

	// Predecir clase
	prediccion := softmaxModel.Predict(Xmat)
	claseSoftmax := prediccion[0]

	// Obtener probabilidades
	probsMat := softmaxModel.PredictProba(Xmat)
	probsRow := probsMat.RawRowView(0)
	fmt.Printf("  • Clase predicha: %d\n", claseSoftmax)
	fmt.Printf("  • Probabilidades: ")
	for i, p := range probsRow {
		fmt.Printf("[%d]=%.3f ", i, p)
	}
	fmt.Println()

	// ========== PASO 4: MAPEO A DIAGNÓSTICO ==========
	fmt.Println("\n[PASO 4] Mapeo a diagnóstico médico")
	diagnostico, existe := clasificacionMedica[claseSoftmax]
	if !existe {
		diagnostico = clasificacionMedica[0] // Default: ninguna enfermedad
	}

	fmt.Printf("  • Enfermedad: %s\n", diagnostico.Enfermedad)
	fmt.Printf("  • Urgencia: %s\n", diagnostico.Urgencia)
	fmt.Printf("  • Crónica: %s\n", diagnostico.Cronica)

	// Determinar flags de pecho y respiración
	pecho := "pecho_no"
	if entrada.redflag_pecho {
		pecho = "pecho_si"
	}
	respiracion := "resp_no"
	if entrada.redflag_respiracion {
		respiracion = "resp_si"
	}

	// ========== PASO 5: PROLOG - OBTENER MEDICAMENTOS CONTRAINDICADOS ==========
	fmt.Println("\n[PASO 5] Obteniendo medicamentos CONTRAINDICADOS desde Prolog")
	medicamentosContraindicados := obtenerMedicamentosContraindicados(
		maquinaProlog,
		diagnostico.Urgencia,
		diagnostico.Enfermedad,
		diagnostico.Cronica,
		pecho,
		respiracion,
	)

	// ========== PASO 6: ANÁLISIS DE RESULTADOS ==========
	fmt.Println("\n[PASO 6] Analizando medicamentos contraindicados")

	// TODOS los medicamentos que devolvió Prolog son contraindicados
	totalContraindicados := len(medicamentosContraindicados)

	fmt.Printf("  • Total medicamentos CONTRAINDICADOS: %d\n", totalContraindicados)

	// ========== PASO 7: GENERAR ADVERTENCIAS ==========
	var advertencias []string

	// Advertencias según la enfermedad
	if diagnostico.Enfermedad != "ninguna" {
		advertencias = append(advertencias,
			fmt.Sprintf("📋 Diagnóstico: %s", diagnostico.Enfermedad))
	}

	// Advertencias según urgencia
	if diagnostico.Urgencia == "alta" {
		advertencias = append(advertencias,
			"🚨 URGENCIA ALTA: Se recomienda atención médica inmediata")
	}

	// Advertencias por red flags
	if featuresTexto.redflag_pecho {
		advertencias = append(advertencias,
			"⚠️ RED FLAG: Dolor o presión en el pecho detectado")
	}
	if featuresTexto.redflag_respiracion {
		advertencias = append(advertencias,
			"⚠️ RED FLAG: Dificultad respiratoria severa detectada")
	}

	// Advertencia sobre medicamentos contraindicados
	if totalContraindicados > 0 {
		advertencias = append(advertencias,
			fmt.Sprintf("⚠️ IMPORTANTE: %d medicamentos están CONTRAINDICADOS para esta enfermedad",
				totalContraindicados))
	}

	// ========== RESPUESTA FINAL ==========
	respuesta := &DiagnosticoResponse{
		EnfermedadDetectada:       diagnostico.Enfermedad,
		NivelUrgencia:             diagnostico.Urgencia,
		ProbabilidadesHuggingFace: probabilidadesHF,
		ClaseSoftmax:              claseSoftmax,
		MedicamentosEvaluados:     medicamentosContraindicados, // Solo los contraindicados
		TotalContraindicados:      totalContraindicados,
		Advertencias:              advertencias,
		TextoRecibido:             req.Texto,
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("DIAGNÓSTICO COMPLETADO")
	fmt.Println(strings.Repeat("=", 60) + "\n")

	return respuesta, nil
}

// ============================================================================
// SERVIDOR HTTP (FIBER)
// ============================================================================

func main() {

	godotenv.Load()
	app := fiber.New(fiber.Config{
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			return c.Status(500).JSON(fiber.Map{
				"error": err.Error(),
			})
		},
	})

	// ========== MIDDLEWARE CORS ==========
	app.Use(func(c *fiber.Ctx) error {
		c.Set("Access-Control-Allow-Origin", "*")
		c.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Set("Access-Control-Allow-Headers", "Content-Type")

		if c.Method() == fiber.MethodOptions {
			return c.SendStatus(fiber.StatusOK)
		}

		return c.Next()
	})

	// ========== INICIALIZACIÓN ==========
	fmt.Println("🚀 Iniciando servidor UniMatch...")

	// Cargar base de conocimiento Prolog
	fmt.Println("📚 Cargando base de conocimiento Prolog...")
	programa := cargarProlog("./prolog/conocimiento.pl")
	maquinaProlog = golog.NewMachine().Consult(programa)
	fmt.Println("✓ Prolog cargado")

	// Intentar cargar modelo Softmax desde disco
	fmt.Println("🧠 Cargando modelo Softmax...")
	if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
		softmaxModel = model
		fmt.Println("✓ Modelo Softmax cargado desde", softmaxModelPath)
	} else {
		fmt.Println("⚠ Modelo Softmax no encontrado. Entrénelo vía /softmax/train")
	}

	// ========== RUTAS ==========

	// Ruta principal
	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"servicio":    "UniMatch Medical Diagnosis API",
			"version":     "3.0",
			"estado":      "activo",
			"descripcion": "Sistema que evalúa medicamentos y detecta contraindicaciones",
			"flujo": []string{
				"1. Análisis de texto (síntomas y red flags)",
				"2. HuggingFace (probabilidades de enfermedades)",
				"3. Softmax (clasificación final)",
				"4. Prolog (entrega TODOS los medicamentos)",
				"5. Evaluación (marca cuáles están contraindicados)",
			},
			"endpoints": []string{
				"POST /diagnostico - Diagnóstico completo con evaluación de medicamentos",
				"POST /softmax/train - Entrenar modelo Softmax",
				"POST /softmax/predict - Predicción con Softmax",
			},
		})
	})

	// ========== ENDPOINT PRINCIPAL: DIAGNÓSTICO ==========
	app.Post("/diagnostico", func(c *fiber.Ctx) error {
		var req DiagnosticoRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{
				"error":   "Error al parsear JSON de entrada",
				"detalle": err.Error(),
			})
		}

		if req.Texto == "" {
			return c.Status(400).JSON(fiber.Map{
				"error": "El campo 'texto' es requerido",
			})
		}

		// Procesar diagnóstico completo
		respuesta, err := procesarDiagnostico(req)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error":   "Error al procesar diagnóstico",
				"detalle": err.Error(),
			})
		}

		return c.JSON(respuesta)
	})

	// ========== ENDPOINT: ENTRENAR SOFTMAX ==========
	app.Post("/softmax/train", func(c *fiber.Ctx) error {
		var req struct {
			X         [][]float64 `json:"x"`
			Y         []int       `json:"y"`
			Lr        float64     `json:"lr"`
			NIter     int         `json:"n_iter"`
			RegLambda float64     `json:"reg_lambda"`
		}

		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Error al parsear entrada"})
		}

		if len(req.X) == 0 || len(req.Y) == 0 {
			return c.Status(400).JSON(fiber.Map{"error": "X e Y son requeridos"})
		}

		if len(req.X) != len(req.Y) {
			return c.Status(400).JSON(fiber.Map{"error": "X e Y deben tener el mismo tamaño"})
		}

		// Valores por defecto
		lr := req.Lr
		if lr == 0 {
			lr = 0.1
		}
		nIter := req.NIter
		if nIter == 0 {
			nIter = 2000
		}
		reg := req.RegLambda
		if reg == 0 {
			reg = 1e-3
		}

		// Convertir a matriz Gonum
		Xmat, err := slice2DToDense(req.X)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
		}

		// Entrenar modelo
		fmt.Println("🎓 Entrenando modelo Softmax...")
		model := algorithms.NewSoftmaxRegression(lr, nIter, reg)
		model.Fit(Xmat, req.Y)
		acc := model.Accuracy(Xmat, req.Y)

		// Guardar modelo globalmente y en disco
		softmaxModel = model
		if err := model.SaveToFile(softmaxModelPath); err != nil {
			fmt.Println("⚠ Error al guardar modelo:", err)
		} else {
			fmt.Println("✓ Modelo guardado en", softmaxModelPath)
		}

		return c.JSON(fiber.Map{
			"mensaje":    "Modelo Softmax entrenado exitosamente",
			"accuracy":   acc,
			"lr":         lr,
			"n_iter":     nIter,
			"reg_lambda": reg,
		})
	})

	// ========== ENDPOINT: PREDICCIÓN SOFTMAX ==========
	app.Post("/softmax/predict", func(c *fiber.Ctx) error {
		var req struct {
			X [][]float64 `json:"x"`
		}

		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Error al parsear entrada"})
		}

		if len(req.X) == 0 {
			return c.Status(400).JSON(fiber.Map{"error": "X es requerido"})
		}

		// Verificar modelo
		if softmaxModel == nil {
			if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
				softmaxModel = model
			} else {
				return c.Status(400).JSON(fiber.Map{
					"error": "Modelo no entrenado. Primero llame a /softmax/train",
				})
			}
		}

		// Convertir y predecir
		Xmat, err := slice2DToDense(req.X)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
		}

		yPred := softmaxModel.Predict(Xmat)
		probsMat := softmaxModel.PredictProba(Xmat)
		probs := denseTo2D(probsMat)

		return c.JSON(fiber.Map{
			"y_pred": yPred,
			"probs":  probs,
		})
	})

	// ========== INICIAR SERVIDOR ==========
	fmt.Println("\n✅ Servidor UniMatch activo en puerto 8080")
	fmt.Println("📡 Endpoints disponibles:")
	fmt.Println("   • GET  /               - Info del servicio")
	fmt.Println("   • POST /diagnostico    - Diagnóstico con contraindicaciones")
	fmt.Println("   • POST /softmax/train  - Entrenar modelo")
	fmt.Println("   • POST /softmax/predict - Predicción directa")
	fmt.Println()

	if err := app.Listen(":8080"); err != nil {
		fmt.Printf("❌ Error al iniciar servidor: %v\n", err)
	}
}
