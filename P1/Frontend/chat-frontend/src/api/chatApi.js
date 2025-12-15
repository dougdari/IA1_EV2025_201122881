/*export async function sendMessage(text) {
  const res = await fetch('http://localhost:8080', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text })
  })

  if (!res.ok) throw new Error('API error')

  return res.json()
}*/

export async function sendMessage(texto) {
  const res = await fetch('http://localhost:8080/diagnostico', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texto })
  })

  if (!res.ok) throw new Error('API error')
  return res.json()
}
