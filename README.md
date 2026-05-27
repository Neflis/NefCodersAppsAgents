# Multi Agent Lab

Sistema multiagente local y asincrono en Python, preparado para crecer hacia una integracion con Ollama sin depender de APIs externas ni frameworks de agentes.

## Objetivo actual

La fase 2 demuestra comunicacion multiagente asincrona con persistencia local y trazabilidad:

- Bus de mensajes con `asyncio`.
- Persistencia SQLite para `messages`, `tasks` y `agent_events`.
- Cola de tareas con cambios de estado persistidos.
- Tres agentes locales: planner, coder y reviewer.
- Logger de eventos de agentes.
- Configuracion mediante `.env`.
- Cliente Ollama con `health_check()`, `generate()`, timeout y manejo de errores.
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
DATABASE_URL=sqlite:///multi_agent_lab.db
```

## Ejecutar demo sin Ollama

```powershell
python -m multi_agent_lab.main --mode demo_mock
```

Este modo no llama a Ollama. El flujo genera un `README.md` de ejemplo dentro de `./workspace/`:

```text
Planner -> Coder -> Reviewer -> FileTool
```

El planner crea una tarea de archivo, el coder genera contenido local simulado, el reviewer valida la propuesta y `FileTool` escribe el archivo solo si la ruta es segura.

## Ejecutar demo con Ollama

```powershell
python -m multi_agent_lab.main --mode demo_ollama
```

Este modo comprueba si Ollama responde en `OLLAMA_BASE_URL`. Si esta disponible, el coder usa el modelo configurado en `OLLAMA_MODEL`. Si no esta disponible, la demo continua con respuesta local simulada.

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
