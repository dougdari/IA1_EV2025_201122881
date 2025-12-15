import { sendMessage } from '../api/chatApi.js'

export function Chat(container) {
  container.innerHTML = `
    <div class="chat">
      <div class="messages" id="messages"></div>

      <div class="input-box">
        <input id="text" placeholder="Escribe un mensaje" />
        <button id="send">Enviar</button>
      </div>
    </div>
  `

  const messages = container.querySelector('#messages')
  const input = container.querySelector('#text')
  const button = container.querySelector('#send')

  button.onclick = async () => {
    if (!input.value) return

    // Mostrar lo que envió el usuario
    addMessage(messages, 'Tú', input.value, 'user')

    button.disabled = true
    button.textContent = '...'

    try {
      const response = await sendMessage(input.value)
      mostrarResultado(messages, response)
    } catch {
      addMessage(messages, 'Error', 'No se pudo conectar', 'bot')
    }

    button.disabled = false
    button.textContent = 'Enviar'
    input.value = ''
  }
}

function addMessage(container, who, text, cls) {
  const p = document.createElement('p')
  p.className = cls
  p.textContent = `${who}: ${text}`
  container.appendChild(p)
  container.scrollTop = container.scrollHeight
}

// Función para mostrar la información específica de la respuesta
function mostrarResultado(container, data) {
  // Texto recibido
  addMessage(container, 'Bot', `Texto recibido: ${data.texto_recibido}`, 'bot')

  // Enfermedad detectada
  addMessage(container, 'Bot', `Enfermedad detectada: ${data.enfermedad_detectada}`, 'bot')

  // Medicamentos con Match > 70%
  const medicamentosAltos = data.medicamentos_evaluados
    .filter(m => m.Match > 70)
    .map(m => `${m.Medicamento} (Match: ${m.Match.toFixed(1)}%)`)

  if (medicamentosAltos.length > 0) {
    addMessage(container, 'Bot', `Medicamentos no recomendados:\n- ${medicamentosAltos.join('\n- ')}`, 'bot')
  } else {
    addMessage(container, 'Bot', 'No hay medicamentos con Match > 70%', 'bot')
  }
}
