package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
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
	a_asma         float32
	a_bronquitis   float32
	a_enfisema     float32
	a_apnea        float32
	a_fibromialgia float32
	a_migranas     float32
	a_reflujo      float32

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

var sintomasKeywords = []string{
	"pecho", "tos", "flema", "silbido", "falta de aire",
	"ahogo", "dificultad para respirar", "opresion", "dolor al respirar",
	"cansancio", "fatiga", "sibilancias", "esputo", "mucosidad",
}

var cronicasKeywords = []string{
	"asma", "epoc", "bronquitis cronica", "fibrosis pulmonar",
	"enfisema", "apnea", "cronico", "cronica", "años de", "desde hace",
}

var pechoKeywords = []string{
	"dolor de pecho", "opresion en el pecho", "presion en el pecho",
	"pecho apretado", "dolor toracico", "dolor intenso en el pecho",
}

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

func analizarTexto(texto string) FeaturesTexto {
	textoLower := strings.ToLower(texto)

	var features FeaturesTexto

	for _, sintoma := range sintomasKeywords {
		if strings.Contains(textoLower, sintoma) {
			features.n_sintomas++
		}
	}

	for _, cronica := range cronicasKeywords {
		if strings.Contains(textoLower, cronica) {
			features.n_cronicas++
			features.tiene_cronicas = true
		}
	}

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

// ============================================================================
// INTEGRACIÓN CON HUGGINGFACE
// ============================================================================

func llamarHuggingFace(texto string) (map[string]float64, error) {
	fmt.Println("Llamando a HuggingFace API...")

	token := os.Getenv("HF_TOKEN")
	if token == "" {
		return nil, fmt.Errorf("la variable de entorno HF_TOKEN no está configurada")
	}

	url := "https://router.huggingface.co/hf-inference/models/PlanTL-GOB-ES/bsc-bio-ehr-es"

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

	var resultado interface{}
	if err := json.Unmarshal(body, &resultado); err != nil {
		return nil, fmt.Errorf("error al parsear JSON: %v", err)
	}

	probabilidades := make(map[string]float64)

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

	fmt.Printf("HuggingFace: %d enfermedades detectadas\n", len(probabilidades))
	return probabilidades, nil
}

// ============================================================================
// EVALUACIÓN DE MEDICAMENTOS CON PROLOG
// ============================================================================

func obtenerMedicamentosContraindicados(
	m golog.Machine,
	urgencia, enfermedad, cronica, pecho, respiracion string,
) []MedicamentoRecomendado {

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

		if urg == urgencia {
			matchCount++
		}
		if enf == enfermedad {
			matchCount++
		}
		if cron == cronica {
			matchCount++
		}
		if pech == pecho {
			matchCount++
		}
		if resp == respiracion {
			matchCount++
		}

		if urgencia == "alta" && respiracion == "resp_si" {
			matchCount++
		}

		if (enfermedad == "asma" || enfermedad == "bronquitis" || enfermedad == "enfisema") &&
			cronica == "cronica_si" && respiracion == "resp_si" {
			matchCount++
		}

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

func procesarDiagnostico(req DiagnosticoRequest) (*DiagnosticoResponse, error) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("INICIANDO DIAGNOSTICO MEDICO")
	fmt.Println(strings.Repeat("=", 60))

	fmt.Println("\n[PASO 1] Analisis de texto del paciente")
	featuresTexto := analizarTexto(req.Texto)
	fmt.Printf("  Sintomas detectados: %d\n", featuresTexto.n_sintomas)
	fmt.Printf("  Enfermedades cronicas: %d\n", featuresTexto.n_cronicas)
	fmt.Printf("  Red flag pecho: %v\n", featuresTexto.redflag_pecho)
	fmt.Printf("  Red flag respiracion: %v\n", featuresTexto.redflag_respiracion)

	fmt.Println("\n[PASO 2] Analisis con HuggingFace (NLP)")
	probabilidadesHF, err := llamarHuggingFace(req.Texto)
	if err != nil {
		return nil, fmt.Errorf("error en HuggingFace: %v", err)
	}

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

	fmt.Println("\n[PASO 3] Clasificacion con modelo Softmax")

	if softmaxModel == nil {
		if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
			softmaxModel = model
			fmt.Println("Modelo cargado desde disco")
		} else {
			return nil, fmt.Errorf("modelo Softmax no disponible. Entrenelo primero")
		}
	}

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

	prediccion := softmaxModel.Predict(Xmat)
	claseSoftmax := prediccion[0]

	probsMat := softmaxModel.PredictProba(Xmat)
	probsRow := probsMat.RawRowView(0)
	fmt.Printf("  Clase predicha: %d\n", claseSoftmax)
	fmt.Printf("  Probabilidades: ")
	for i, p := range probsRow {
		fmt.Printf("[%d]=%.3f ", i, p)
	}
	fmt.Println()

	fmt.Println("\n[PASO 4] Mapeo a diagnostico medico")
	diagnostico, existe := clasificacionMedica[claseSoftmax]
	if !existe {
		diagnostico = clasificacionMedica[0]
	}

	fmt.Printf("  Enfermedad: %s\n", diagnostico.Enfermedad)
	fmt.Printf("  Urgencia: %s\n", diagnostico.Urgencia)
	fmt.Printf("  Cronica: %s\n", diagnostico.Cronica)

	pecho := "pecho_no"
	if entrada.redflag_pecho {
		pecho = "pecho_si"
	}
	respiracion := "resp_no"
	if entrada.redflag_respiracion {
		respiracion = "resp_si"
	}

	fmt.Println("\n[PASO 5] Obteniendo medicamentos contraindicados desde Prolog")
	medicamentosContraindicados := obtenerMedicamentosContraindicados(
		maquinaProlog,
		diagnostico.Urgencia,
		diagnostico.Enfermedad,
		diagnostico.Cronica,
		pecho,
		respiracion,
	)

	fmt.Println("\n[PASO 6] Analizando medicamentos contraindicados")
	totalContraindicados := len(medicamentosContraindicados)
	fmt.Printf("  Total medicamentos contraindicados: %d\n", totalContraindicados)

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
	if featuresTexto.redflag_respiracion {
		advertencias = append(advertencias,
			"RED FLAG: Dificultad respiratoria severa detectada")
	}

	if totalContraindicados > 0 {
		advertencias = append(advertencias,
			fmt.Sprintf("IMPORTANTE: %d medicamentos estan contraindicados para esta enfermedad",
				totalContraindicados))
	}

	respuesta := &DiagnosticoResponse{
		EnfermedadDetectada:       diagnostico.Enfermedad,
		NivelUrgencia:             diagnostico.Urgencia,
		ProbabilidadesHuggingFace: probabilidadesHF,
		ClaseSoftmax:              claseSoftmax,
		MedicamentosEvaluados:     medicamentosContraindicados,
		TotalContraindicados:      totalContraindicados,
		Advertencias:              advertencias,
		TextoRecibido:             req.Texto,
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("DIAGNOSTICO COMPLETADO")
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

	app.Use(func(c *fiber.Ctx) error {
		c.Set("Access-Control-Allow-Origin", "*")
		c.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Set("Access-Control-Allow-Headers", "Content-Type")

		if c.Method() == fiber.MethodOptions {
			return c.SendStatus(fiber.StatusOK)
		}

		return c.Next()
	})

	fmt.Println("Iniciando servidor UniMatch...")

	fmt.Println("Cargando base de conocimiento Prolog...")
	programa := cargarProlog("./prolog/conocimiento.pl")
	maquinaProlog = golog.NewMachine().Consult(programa)
	fmt.Println("Prolog cargado")

	fmt.Println("Cargando modelo Softmax...")
	if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
		softmaxModel = model
		fmt.Println("Modelo Softmax cargado desde", softmaxModelPath)
	} else {
		fmt.Println("Modelo Softmax no encontrado. Entrenelo via /softmax/train")
	}

	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"servicio":    "UniMatch Medical Diagnosis API",
			"version":     "3.0",
			"estado":      "activo",
			"descripcion": "Sistema que evalua medicamentos y detecta contraindicaciones",
			"flujo": []string{
				"1. Analisis de texto (sintomas y red flags)",
				"2. HuggingFace (probabilidades de enfermedades)",
				"3. Softmax (clasificacion final)",
				"4. Prolog (entrega todos los medicamentos)",
				"5. Evaluacion (marca cuales estan contraindicados)",
			},
			"endpoints": []string{
				"POST /diagnostico - Diagnostico completo con evaluacion de medicamentos",
				"POST /softmax/train - Entrenar modelo Softmax",
				"POST /softmax/predict - Prediccion con Softmax",
			},
		})
	})

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

		respuesta, err := procesarDiagnostico(req)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error":   "Error al procesar diagnostico",
				"detalle": err.Error(),
			})
		}

		///RPA

		exePath, _ := os.Getwd()

		cmd := exec.Command(
			"powershell",
			"-NoProfile",
			"-ExecutionPolicy", "Bypass",
			"-File", "rpa.ps1",
		)

		cmd.Dir = exePath

		out, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Println("Error ejecutando PowerShell:", err)
		}

		fmt.Println(string(out))

		///

		return c.JSON(respuesta)
	})

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

		Xmat, err := slice2DToDense(req.X)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
		}

		fmt.Println("Entrenando modelo Softmax...")
		model := algorithms.NewSoftmaxRegression(lr, nIter, reg)
		model.Fit(Xmat, req.Y)
		acc := model.Accuracy(Xmat, req.Y)

		softmaxModel = model
		if err := model.SaveToFile(softmaxModelPath); err != nil {
			fmt.Println("Error al guardar modelo:", err)
		} else {
			fmt.Println("Modelo guardado en", softmaxModelPath)
		}

		return c.JSON(fiber.Map{
			"mensaje":    "Modelo Softmax entrenado exitosamente",
			"accuracy":   acc,
			"lr":         lr,
			"n_iter":     nIter,
			"reg_lambda": reg,
		})
	})

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

		if softmaxModel == nil {
			if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
				softmaxModel = model
			} else {
				return c.Status(400).JSON(fiber.Map{
					"error": "Modelo no entrenado. Primero llame a /softmax/train",
				})
			}
		}

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

	fmt.Println("\nServidor UniMatch activo en puerto 8080")
	fmt.Println("Endpoints disponibles:")
	fmt.Println("   GET  /")
	fmt.Println("   POST /diagnostico")
	fmt.Println("   POST /softmax/train")
	fmt.Println("   POST /softmax/predict")
	fmt.Println()

	if err := app.Listen(":8080"); err != nil {
		fmt.Printf("Error al iniciar servidor: %v\n", err)
	}
}
