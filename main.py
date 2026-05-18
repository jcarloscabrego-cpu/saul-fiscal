"""
main.py — Backend de Saúl
FastAPI + Claude API
Corre con: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import anthropic
import os

app = FastAPI(title="Saúl API")

# CORS para que la app web pueda llamar al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente de Anthropic — lee la API key de variable de entorno
cliente = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


# ─── Base de datos simulada (en producción sería PostgreSQL) ───────────────────

USUARIOS = {
    "GACM891014HG3": {
        "nombre": "Karla García Cruz",
        "rfc": "GACM891014HG3",
        "regimen": "RESICO",
        "zona": "general",
        "ingreso_acumulado_anio": 187_400,
        "declaraciones_tardias": 0,
        "tasa_isr": 1.0,
        "tasa_iva": 8.0,
        "tope_anual": 3_500_000,
        "historial": [
            {"mes": "Octubre 2025",  "total": 1180, "estado": "Pagado"},
            {"mes": "Septiembre 2025", "total": 980,  "estado": "Pagado"},
            {"mes": "Agosto 2025",   "total": 1420, "estado": "Pagado"},
        ],
        "mes_actual": {
            "mes": "Noviembre 2025",
            "ingresos_cobrados": 23_400,
            "ingresos_efectivo": 2_000,
            "isr_neto": 0,
            "iva_neto": 1_125,
            "saldo_favor_isr": 546,
            "facturas_emitidas": 8,
            "facturas_cobradas": 6,
        }
    }
}

HISTORIAL_CHAT = {}  # guarda conversaciones por RFC


# ─── Modelos ───────────────────────────────────────────────────────────────────

class MensajeChat(BaseModel):
    rfc: str
    mensaje: str


class RespuestaChat(BaseModel):
    respuesta: str
    sugerencias: Optional[list[str]] = None


# ─── Sistema prompt de Saúl ────────────────────────────────────────────────────

def construir_system_prompt(usuario: dict) -> str:
    mes = usuario["mes_actual"]
    hist = "\n".join([
        f"  - {h['mes']}: ${h['total']:,} ({h['estado']})"
        for h in usuario["historial"]
    ])

    return f"""Eres Saúl, el asesor fiscal personal de {usuario['nombre']}.

PERFIL FISCAL DEL USUARIO:
- Nombre: {usuario['nombre']}
- RFC: {usuario['rfc']}
- Régimen: {usuario['regimen']}
- Zona fiscal: {usuario['zona']}
- Ingresos acumulados este año: ${usuario['ingreso_acumulado_anio']:,} MXN
- Tasa ISR actual: {usuario['tasa_isr']}%
- Tasa IVA: {usuario['tasa_iva']}%
- Tope anual RESICO: ${usuario['tope_anual']:,} MXN
- Declaraciones tardías este año: {usuario['declaraciones_tardias']}

MES ACTUAL ({mes['mes']}):
- Facturas emitidas: {mes['facturas_emitidas']} ({mes['facturas_cobradas']} cobradas)
- Ingresos cobrados: ${mes['ingresos_cobrados']:,} MXN
- Ingresos en efectivo: ${mes['ingresos_efectivo']:,} MXN
- ISR a pagar: ${mes['isr_neto']:,} MXN
- IVA a pagar: ${mes['iva_neto']:,} MXN
- Saldo a favor ISR (para diciembre): ${mes['saldo_favor_isr']:,} MXN

HISTORIAL RECIENTE:
{hist}

CÓMO ERES:
- Hablas como un amigo que estudió contabilidad — cercano, directo, sin jargon
- Nunca usas términos fiscales sin explicarlos inmediatamente en lenguaje simple
- Siempre terminas con una pregunta o una acción concreta
- Cuando el usuario tiene saldo a favor o está pagando de más, lo dices claro
- Si algo es urgente (declaración próxima a vencer), lo dices primero
- Nunca asustas al usuario — el SAT es manejable, tú estás para eso
- Eres breve: máximo 3-4 oraciones por respuesta
- Usas el nombre del usuario ocasionalmente para que se sienta personal

LÍMITES:
- Solo asesoras sobre situación fiscal mexicana de personas físicas
- Si preguntan algo muy complejo que requiere revisión manual, dices que lo revisas con detalle
- No inventas números — si no tienes el dato exacto, dilo y ofrece calcularlo
- Nunca presentas declaraciones sin confirmación explícita del usuario"""


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=RespuestaChat)
async def chat(body: MensajeChat):
    usuario = USUARIOS.get(body.rfc)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Historial de conversación del usuario (memoria dentro de la sesión)
    if body.rfc not in HISTORIAL_CHAT:
        HISTORIAL_CHAT[body.rfc] = []

    historial = HISTORIAL_CHAT[body.rfc]
    historial.append({"role": "user", "content": body.mensaje})

    # Llamada a Claude API
    respuesta = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=construir_system_prompt(usuario),
        messages=historial
    )

    texto = respuesta.content[0].text

    # Guardar respuesta en historial
    historial.append({"role": "assistant", "content": texto})

    # Limitar historial a 20 mensajes para no crecer infinito
    if len(historial) > 20:
        HISTORIAL_CHAT[body.rfc] = historial[-20:]

    # Sugerencias contextuales según el mensaje
    sugerencias = generar_sugerencias(body.mensaje, usuario)

    return RespuestaChat(respuesta=texto, sugerencias=sugerencias)


@app.get("/usuario/{rfc}/resumen")
async def resumen(rfc: str):
    usuario = USUARIOS.get(rfc)
    if not usuario:
        raise HTTPException(status_code=404, detail="No encontrado")
    return usuario


@app.delete("/chat/{rfc}/limpiar")
async def limpiar_chat(rfc: str):
    """Limpia el historial de conversación — útil para nueva sesión"""
    HISTORIAL_CHAT.pop(rfc, None)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "modelo": "claude-sonnet-4-6"}


# ─── Sugerencias contextuales ──────────────────────────────────────────────────

def generar_sugerencias(mensaje: str, usuario: dict) -> list[str]:
    """Genera botones de respuesta rápida según el contexto del mensaje"""
    msg = mensaje.lower()
    mes = usuario["mes_actual"]

    if any(w in msg for w in ["declarar", "presentar", "pagar"]):
        return ["¿Cuánto pago exactamente?", "¿Cuándo vence?", "¿Cómo pago?"]
    if any(w in msg for w in ["deducir", "deducción", "gasto"]):
        return ["¿Qué gastos puedo deducir?", "¿Necesito factura?", "¿Y los gastos en efectivo?"]
    if any(w in msg for w in ["acumulado", "llevo", "año"]):
        return ["¿Cuánto me falta para el tope?", "¿Cuándo sube mi tasa?"]
    if mes["iva_neto"] > 0:
        return ["¿Por qué ese monto de IVA?", "Quiero declarar ahora", "¿Puedo pagarlo después?"]
    return ["¿Cuándo es mi próxima declaración?", "¿Tengo saldo a favor?", "¿Qué necesito para declarar?"]
