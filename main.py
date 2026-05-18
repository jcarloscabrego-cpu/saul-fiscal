from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
cliente = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
historial_chats = {}

class MensajeChat(BaseModel):
    rfc: str
    mensaje: str
    regimen: str
    nombre: str
    ingreso_acumulado: Optional[float] = 0
    isr_neto: Optional[float] = 0
    iva_neto: Optional[float] = 0
    saldo_favor: Optional[float] = 0

class RespuestaChat(BaseModel):
    respuesta: str
    sugerencias: Optional[list] = []

def system_prompt(u):
    return f"""Eres Saúl, asesor fiscal de {u.nombre}. RFC: {u.rfc}. Régimen: {u.regimen}.
Ingresos acumulados: ${u.ingreso_acumulado:,.0f}. ISR: ${u.isr_neto:,.2f}. IVA: ${u.iva_neto:,.2f}.
Habla como amigo contador. Sin jerga. Máximo 3 oraciones. Termina con pregunta o acción."""

@app.get("/")
def root():
    return {"status": "Saúl API ok", "version": "0.1"}

@app.post("/chat", response_model=RespuestaChat)
async def chat(body: MensajeChat):
    if body.rfc not in historial_chats:
        historial_chats[body.rfc] = []
    hist = historial_chats[body.rfc]
    hist.append({"role":"user","content":body.mensaje})
    try:
        resp = cliente.messages.create(model="claude-sonnet-4-6", max_tokens=300, system=system_prompt(body), messages=hist)
        texto = resp.content[0].text
        hist.append({"role":"assistant","content":texto})
        if len(hist) > 20:
            historial_chats[body.rfc] = hist[-20:]
    except Exception as e:
        texto = "Problema de conexión. Intenta de nuevo."
    sugs = ["¿Cuánto llevo acumulado?","¿Tengo saldo a favor?","¿Cuándo es mi anual?"]
    return RespuestaChat(respuesta=texto, sugerencias=sugs)

@app.get("/health")
def health():
    return {"status":"ok"}
