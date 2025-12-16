import { sendMessage } from '../api/chatApi.js'

export function Chat(container) {
  container.innerHTML = `
    <style>
      .chat {
        display: flex;
        flex-direction: column;
        height: 520px;
        width: 420px;
        border: 2px solid #444;
        border-radius: 12px;
        overflow: hidden;
        font-family: Arial, sans-serif;
        background-color: #1e1e1e;
        color: #fff;
      }

      .messages {
        flex: 1;
        padding: 12px;
        overflow-y: auto;
        background-color: #2c2c2c;
      }

      .messages p {
        margin: 6px 0;
        padding: 10px 14px;
        border-radius: 12px;
        max-width: 95%;
        word-wrap: break-word;
        white-space: pre-line;
      }

      .user {
        background-color: #4caf50;
        align-self: flex-end;
      }

      .bot {
        background-color: #444;
        align-self: flex-start;
      }

      .title {
        font-weight: bold;
        color: #ffd54f;
      }

      .input-box {
        display: flex;
        padding: 10px;
        gap: 8px;
        border-top: 1px solid #444;
        background-color: #1e1e1e;
      }

      .input-box input {
        flex: 1;
        padding: 14px 16px;
        border-radius: 20px;
        border: none;
        outline: none;
        font-size: 15px;
        background-color: #333;
        color: #fff;
      }

      .input-box button {
        padding: 0 22px;
        border-radius: 20px;
        border: none;
        background-color: #4caf50;
        color: #fff;
        cursor: pointer;
        font-weight: bold;
      }

      .input-box button:disabled {
        background-color: #777;
      }
    </style>

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

  const enviar = async () => {
    if (!input.value) return

    addMessage(messages, input.value, 'user')

    button.disabled = true
    button.textContent = '...'

    try {
      const response = await sendMessage(input.value)
      mostrarResultado(messages, response)
    } catch {
      addMessage(messages, 'Error de conexión', 'bot')
    }

    button.disabled = false
    button.textContent = 'Enviar'
    input.value = ''
    input.focus()
  }

  button.onclick = enviar

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault()
      enviar()
    }
  })
}

function addMessage(container, text, cls) {
  const p = document.createElement('p')
  p.className = cls
  p.textContent = text
  container.appendChild(p)
  container.scrollTop = container.scrollHeight
}

function addTitle(container, title, text) {
  const p = document.createElement('p')
  p.className = 'bot'
  p.innerHTML = `<span class="title">${title}:</span>\n${text}`
  container.appendChild(p)
  container.scrollTop = container.scrollHeight
}

function mostrarResultado(container, data) {
  addTitle(container, 'Texto recibido', data.texto_recibido)
  addTitle(container, 'Enfermedad detectada', data.enfermedad_detectada)
  addTitle(container, 'Nivel de urgencia', data.nivel_urgencia)

  const enfermedad = data.enfermedad_detectada

  const medicamentosFiltrados = data.medicamentos_evaluados
    .filter(m => m.Match > 70 && m.Enfermedad === enfermedad)
    .map(m => `• ${m.Medicamento} (${m.Match.toFixed(1)}%)`)

  if (medicamentosFiltrados.length > 0) {
    addTitle(
      container,
      'Medicamentos no recomendados',
      medicamentosFiltrados.join('\n')
    )
  } else {
    addTitle(
      container,
      'Medicamentos no recomendados',
      'No se encontraron medicamentos con Match > 70% para esta enfermedad'
    )
  }
}
