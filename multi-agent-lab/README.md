# Multi Agent Lab

Sistema multiagente local y asíncrono en Python, preparado para crecer hacia una integración con Ollama sin depender de APIs externas ni frameworks de agentes.

## Objetivo de esta fase

Esta primera fase solo demuestra comunicación multiagente asíncrona:

- Bus de mensajes con `asyncio`.
- Cola de tareas básica.
- Tres agentes locales: planner, coder y reviewer.
- Cliente de Ollama preparado como interfaz futura, sin llamadas reales obligatorias.
- Tests básicos para el bus y la cola.

No se escriben archivos generados por agentes, no se ejecutan comandos del sistema y no se usa Docker.

## Requisitos

- Python 3.11 o superior.
- Entorno local con `pytest` para ejecutar tests.

## Instalación

```powershell
cd multi-agent-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Ejecutar la demo

```powershell
python -m multi_agent_lab.main
```

La demo arranca un planner, un coder y un reviewer. El planner crea una tarea de ejemplo, el coder devuelve una respuesta simulada y el reviewer publica una revisión simulada.

## Ejecutar tests

```powershell
pytest
```

## Estructura

```text
src/multi_agent_lab/
  core/       Modelos, bus de mensajes, tareas, cola y memoria simple.
  agents/     Agentes asíncronos.
  llm/        Cliente preparado para Ollama.
  tools/      Espacio reservado para herramientas futuras.
  config/     Configuración local.
```
