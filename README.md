# Multi Agent Lab

Sistema multiagente local y asincrono en Python, preparado para crecer hacia una integracion con Ollama sin depender de APIs externas ni frameworks de agentes.

## Objetivo actual

La fase actual demuestra una red multiagente autonoma basada en eventos y un grafo dinamico de tareas:

- Bus de mensajes con `asyncio`.
- Agentes desacoplados que escuchan eventos y publican nuevos eventos.
- `correlation_id`, `causation_id` y `metadata` en cada mensaje.
- `TaskGraph` para descomponer objetivos en tareas con dependencias.
- Modelo de capabilities para que los trabajadores reclamen tareas compatibles.
- Persistencia SQLite para `messages`, `tasks` y `agent_events`.
- Persistencia de snapshots del grafo en `task_graphs`.
- Agentes locales: planner, coder, reviewer, file, tester, coordinator y supervisor.
- Logger de eventos de agentes.
- Configuracion mediante `.env`.
- Cliente Ollama con `health_check()`, `generate()`, timeout y manejo de errores.
- Razonamiento LLM estructurado con `PromptTemplate`, `AgentContextBuilder` y `LLMDecision`.
- Workspace seguro en `./workspace/` para operaciones controladas de archivos.

No se ejecutan comandos del sistema, no se usa Docker y no se permite borrado de archivos.

## Requisitos

- Python 3.11 o superior.
- Entorno local con `pytest` para ejecutar tests.
- Ollama local solo si quieres usar `demo_ollama`.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Configuracion

Copia `.env.example` a `.env` y ajusta los valores si lo necesitas:

```powershell
Copy-Item .env.example .env
```

Variables disponibles:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_MODEL_PLANNER=llama3.2
OLLAMA_MODEL_CODER=llama3.2
OLLAMA_MODEL_REVIEWER=llama3.2
OLLAMA_TIMEOUT_SECONDS=10
USE_MOCK_LLM=true
DATABASE_URL=sqlite:///multi_agent_lab.db
```

Para usar Ollama real:

```text
USE_MOCK_LLM=false
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_PLANNER=llama3.2
OLLAMA_MODEL_CODER=llama3.2
OLLAMA_MODEL_REVIEWER=llama3.2
```

Modelos recomendados para pruebas locales: `llama3.2`, `qwen2.5-coder`, `mistral`.
El modo mock (`USE_MOCK_LLM=true`) no llama a Ollama y devuelve JSON determinista.

## Ejecutar demo sin Ollama

```powershell
python -m multi_agent_lab.main --mode demo_mock
```

Este modo no llama a Ollama. La demo publica `GOAL_SUBMITTED` y la red genera un `README.md` de ejemplo dentro de `./workspace/`.

Objetivo de demo:

```text
Crear una pequena documentacion README para una app TODO
```

El planner descompone el objetivo en un `TaskGraph`:

```text
1. Crear borrador README        capability: coding
2. Revisar README               capability: reviewing
3. Escribir README en workspace capability: file_write
4. Validar existencia           capability: testing_mock
```

Flujo de eventos:

```text
GOAL_SUBMITTED
  -> PlannerAgent publica GOAL_DECOMPOSED, TASK_GRAPH_UPDATED y TASK_READY
  -> Workers compatibles publican TASK_CLAIMED
  -> Workers publican TASK_COMPLETED o TASK_FAILED
  -> TaskCoordinatorAgent libera dependientes con TASK_READY
  -> FileAgent escribe con FileTool
  -> TesterAgent simula validacion con TEST_PASSED
```

Ningun agente llama directamente a otro agente. Todos reaccionan de forma autonoma a eventos del `MessageBus`. `SupervisorAgent` escucha `*` y puede publicar `WORKFLOW_HALTED` si detecta demasiados eventos, retries o errores repetidos.

## Ejecutar demo con Ollama

```powershell
python -m multi_agent_lab.main --mode demo_ollama
```

Este modo comprueba si Ollama responde en `OLLAMA_BASE_URL`. Si esta disponible, el coder usa el modelo configurado en `OLLAMA_MODEL`. Si no esta disponible, la demo continua con respuesta local simulada.

## CLI runtime

Ejecutar un objetivo con LLM mock:

```powershell
python -m multi_agent_lab run --goal "Crear README para app TODO" --mock
```

Ejecutar un objetivo multi-archivo:

```powershell
python -m multi_agent_lab run --goal "Crea una pequena API Flask TODO" --mock
```

Resultado esperado en `workspace/`:

```text
app.py
requirements.txt
README.md
```

Ejecutar un objetivo con Ollama real:

```powershell
python -m multi_agent_lab run --goal "Crear README para app TODO" --ollama
```

Usar un workspace sandbox especifico:

```powershell
python -m multi_agent_lab run --goal "Crear README para app TODO" --mock --workspace .\workspace
```

El runtime publica `WORKFLOW_STARTED`, envia `GOAL_SUBMITTED` y espera hasta `WORKFLOW_COMPLETED`, `WORKFLOW_HALTED` o `WORKFLOW_TIMEOUT`. Al terminar muestra objetivo, tareas completadas, tareas fallidas, archivos creados, duracion y estado final.

Para objetivos multi-archivo, `FileAwarenessService` lista y lee archivos relevantes dentro del workspace para que los agentes mantengan coherencia entre `app.py`, `requirements.txt` y `README.md`. No lee fuera del sandbox ni archivos por encima del limite seguro.

## Workspace seguro

Todas las operaciones de archivos estan limitadas a:

```text
./workspace/
```

La capa de seguridad esta separada en dos piezas:

- `WorkspaceManager`: crea el workspace, resuelve rutas absolutas seguras e impide path traversal, rutas absolutas externas y segmentos peligrosos como `.git` o `.env`.
- `FileTool`: permite `read_file`, `write_file`, `append_file`, `list_files` y `exists` solo sobre rutas validadas por `WorkspaceManager`.

Limites actuales:

- Extensiones permitidas: `.py`, `.md`, `.txt`, `.json`, `.yaml`, `.yml`.
- Tamano maximo por archivo: 1 MB.
- No hay borrado de archivos.
- No hay ejecucion de comandos.
- No hay acceso libre al sistema de archivos.

## TaskGraph y capabilities

Cada goal crea un `TaskGraph` identificado por `correlation_id`. Cada `TaskNode` contiene:

- dependencias
- prioridad
- estado: `PENDING`, `READY`, `IN_PROGRESS`, `BLOCKED`, `COMPLETED`, `FAILED`, `CANCELLED`
- owner opcional
- numero de retries
- timestamps
- `required_capability`

Los agentes trabajadores no reciben tareas asignadas. Escuchan `TASK_READY`, revisan `required_capability` y publican `TASK_CLAIMED` si son compatibles. El coordinator actualiza el grafo y publica nuevas tareas listas cuando sus dependencias se completan.

## Razonamiento LLM

Los agentes de planificacion, codigo y revision usan prompts estructurados:

- `PromptTemplate`: identidad, capabilities, restricciones, contexto y salida JSON esperada.
- `AgentContextBuilder`: objetivo, tarea actual, resumen del grafo, eventos recientes y archivos del workspace.
- `LLMDecision`: `action`, `reasoning_summary`, `confidence`, `events_to_publish`, `task_updates`, `content`.

Si Ollama no esta disponible o `USE_MOCK_LLM=true`, se usa modo mock. Si el modelo devuelve JSON invalido, el agente cae a una decision determinista segura.

## Ejecutar tests

```powershell
pytest
```

## Estructura

```text
src/multi_agent_lab/
  core/       Modelos, bus de mensajes, tareas, cola, SQLite y eventos.
  agents/     Agentes asincronos.
  llm/        Cliente Ollama.
  tools/      Herramientas seguras restringidas al workspace.
  config/     Configuracion local.
tests/        Tests basicos y de persistencia.
workspace/    Archivos generados por demos o agentes, ignorados por Git.
```
