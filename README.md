# Multi Agent Lab

Sistema multiagente local y asincrono en Python, preparado para crecer hacia una integracion con Ollama sin depender de APIs externas ni frameworks de agentes.

## Objetivo actual

La fase actual demuestra una red multiagente autonoma basada en eventos y un grafo dinamico de tareas:

- Bus de mensajes con `asyncio`.
- Agentes desacoplados que escuchan eventos y publican nuevos eventos.
- `correlation_id`, `causation_id` y `metadata` en cada mensaje.
- `TaskGraph` para descomponer objetivos en tareas con dependencias.
- `ProjectSpec` para convertir objetivos vagos en una especificacion funcional antes de planificar.
- Modelo de capabilities para que los trabajadores reclamen tareas compatibles.
- Persistencia SQLite para `messages`, `tasks` y `agent_events`.
- Persistencia de snapshots del grafo en `task_graphs`.
- Agentes locales: planner, coder, reviewer, file, tester, coordinator y supervisor.
- Logger de eventos de agentes.
- Configuracion mediante `.env`.
- Cliente Ollama con `health_check()`, `generate()`, timeout y manejo de errores.
- Razonamiento LLM estructurado con `PromptTemplate`, `AgentContextBuilder` y `LLMDecision`.
- Workspace seguro en `./workspace/` para operaciones controladas de archivos.
- `ProjectMemory` semantica persistida para mantener coherencia sin enviar todo el historial al LLM.
- Compresion de contexto y reduccion de ruido de eventos repetidos.
- Salidas LLM JSON robustas con extraccion, reparacion simple, schemas y trazas en `workspace/.traces/`.

La ejecucion de comandos reales esta desactivada por defecto. Cuando se activa explicitamente, solo se permite una whitelist estricta dentro del workspace. No se usa Docker, no se usa git automatico y no se permite borrado de archivos.

## Goal-to-Spec

Antes de que `PlannerAgent` descomponga un objetivo, `SpecAgent` convierte el texto del usuario en un `ProjectSpec` estructurado y lo guarda como JSON en:

```text
workspace/.spec/project_spec.json
```

Flujo:

```text
GOAL_SUBMITTED
  -> SPEC_REQUESTED
  -> SPEC_GENERATED
  -> SPEC_APPROVED
  -> GOAL_DECOMPOSED
```

Para objetivos vagos no se pregunta al usuario todavia: se asume un MVP razonable. Por ejemplo, "Hazme una web para registrar mis ventas de impresion 3D" genera entidades `Product`, `Customer`, `Sale`, `SaleItem` y `Payment`, pantallas como `Dashboard`, `Products`, `Customers`, `Sales` y `New Sale`, y validaciones como `quantity > 0` y `payment status required`.

Esta fase no genera todavia una aplicacion fullstack completa para specs genericas; solo prepara la especificacion y la integra con el planner.

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

Ejecutar el baseline Spring Boot minimo:

```powershell
python -m multi_agent_lab run --goal "Crea una API Spring Boot minima con tests basicos" --mock
```

Resultado esperado en `workspace/`:

```text
pom.xml
src/main/java/com/example/demo/DemoApplication.java
src/main/java/com/example/demo/HealthController.java
src/test/java/com/example/demo/HealthControllerTest.java
README.md
```

Si usas `--allow-execution` y el workspace contiene `pom.xml`, `TesterExecutionAgent` ejecuta `mvn test` mediante la whitelist estricta. La ejecucion real requiere Maven y Java 17 instalados en la maquina.

Ejecutar el baseline Spring Boot CRUD de usuarios:

```powershell
python -m multi_agent_lab run --goal "Crea una API Spring Boot CRUD de usuarios con tests basicos" --mock
```

Resultado esperado en `workspace/`:

```text
pom.xml
src/main/java/com/example/demo/DemoApplication.java
src/main/java/com/example/demo/user/User.java
src/main/java/com/example/demo/user/UserController.java
src/main/java/com/example/demo/user/UserService.java
src/test/java/com/example/demo/user/UserControllerTest.java
README.md
```

Esta vertical usa almacenamiento en memoria con `Map<Long, User>` y expone `GET /users`, `GET /users/{id}`, `POST /users` y `DELETE /users/{id}`. No usa base de datos, JPA ni Lombok. Con `--allow-execution`, tambien se valida mediante `mvn test`.

Ejecutar el baseline Angular minimo:

```powershell
python -m multi_agent_lab run --goal "Crea una aplicación Angular mínima" --mock
```

Resultado esperado en `workspace/`:

```text
package.json
angular.json
tsconfig.json
src/main.ts
src/app/app.component.ts
src/app/app.component.html
```

Esta vertical usa Angular standalone 17+, un unico componente y muestra `Angular Works`. Con `--allow-execution`, el runtime ejecuta `npm install` y despues `npm run build` mediante whitelist estricta.

Ejecutar un objetivo con Ollama real:

```powershell
python -m multi_agent_lab run --goal "Crear README para app TODO" --ollama
```

Ejecutar validacion real controlada dentro del workspace:

```powershell
python -m multi_agent_lab run --goal "Crea una pequena API Flask TODO" --ollama --timeout 600 --allow-execution
```

Puedes ajustar los limites del supervisor y del auto-fix:

```powershell
python -m multi_agent_lab run --goal "Crea una pequena API Flask TODO con tests basicos" --ollama --timeout 900 --allow-execution --max-events 300 --max-fix-attempts 2
```

Usar un workspace sandbox especifico:

```powershell
python -m multi_agent_lab run --goal "Crear README para app TODO" --mock --workspace .\workspace
```

El runtime publica `WORKFLOW_STARTED`, envia `GOAL_SUBMITTED` y espera hasta `WORKFLOW_COMPLETED`, `WORKFLOW_HALTED` o `WORKFLOW_TIMEOUT`. Al terminar muestra objetivo, tareas completadas, tareas fallidas, archivos creados, duracion y estado final.

Para objetivos multi-archivo, `FileAwarenessService` lista y lee archivos relevantes dentro del workspace para que los agentes mantengan coherencia entre `app.py`, `requirements.txt` y `README.md`. No lee fuera del sandbox ni archivos por encima del limite seguro.

El resumen final incluye metricas LLM: respuestas JSON validas, fallos, fallbacks deterministas y latencia media.

Por defecto, la CLI muestra una salida limpia y resume los eventos. Para ver todos los logs de agentes y supervisor:

```powershell
python -m multi_agent_lab run --goal "Crea una pequena API Flask TODO" --mock --verbose
```

## Workspace seguro

Todas las operaciones de archivos estan limitadas a:

```text
./workspace/
```

La capa de seguridad esta separada en dos piezas:

- `WorkspaceManager`: crea el workspace, resuelve rutas absolutas seguras e impide path traversal, rutas absolutas externas y segmentos peligrosos como `.git` o `.env`.
- `FileTool`: permite `read_file`, `write_file`, `append_file`, `list_files` y `exists` solo sobre rutas validadas por `WorkspaceManager`.

Limites actuales:

- Extensiones permitidas: `.py`, `.md`, `.txt`, `.json`, `.yaml`, `.yml`, `.xml`, `.java`, `.ts`, `.html`.
- Tamano maximo por archivo: 1 MB.
- No hay borrado de archivos.
- No hay ejecucion de comandos salvo que uses `--allow-execution`.
- No hay acceso libre al sistema de archivos.

## Ejecucion controlada

`--allow-execution` activa `TesterExecutionAgent` y `CommandTool`. Esta fase existe para validar proyectos generados, pero sigue una politica de seguridad cerrada:

- `shell=False` obligatorio.
- `cwd` siempre es el workspace.
- timeout configurable en la herramienta.
- stdout y stderr se truncan.
- rutas fuera del workspace se bloquean.
- argumentos con tokens sospechosos se rechazan.

Comandos permitidos inicialmente:

```text
python app.py
pytest
pip check
mvn test
npm install
npm run build
```

Comandos bloqueados:

```text
cmd, powershell, bash, curl, wget, rm, del, git, npm, docker
```

La ejecucion se comunica por eventos: `TEST_EXECUTION_REQUESTED`, `TEST_EXECUTION_STARTED`, `TEST_EXECUTION_PASSED` y `TEST_EXECUTION_FAILED`. Si falla, el resultado incluye salida truncada y feedback sobre imports faltantes, errores de sintaxis o timeout. El coordinator mantiene un maximo de 2 retries automaticos antes de bloquear la tarea.

`pytest` se ejecuta con `cwd=workspace` y `PYTHONPATH` apuntando al workspace para que proyectos Python simples puedan importar modulos locales como `app.py` desde `tests/test_app.py`.

Para proyectos Maven, solo se permite `mvn test`. Otros goals como `mvn clean install`, `mvn package` o cualquier combinacion adicional quedan bloqueados por `CommandTool`.

Para proyectos npm, solo se permite `npm install` y `npm run build`. Otros comandos como `npm test`, `npm start`, `npm run dev` o scripts arbitrarios quedan bloqueados.

## Auto-fix tras fallos

Cuando una ejecucion controlada falla, el sistema no detiene el workflow de inmediato. `TaskCoordinatorAgent` convierte `TEST_EXECUTION_FAILED` en una tarea de correccion `coding` y publica `FIX_REQUESTED`.

Flujo actual:

```text
TEST_EXECUTION_FAILED
  -> FIX_REQUESTED
  -> CoderAgent publica FIX_PROPOSED
  -> FileAgent aplica el cambio y publica FIX_APPLIED
  -> TesterExecutionAgent publica RETEST_REQUESTED
  -> pytest se ejecuta otra vez
```

Si el fix modifica `requirements.txt` con estrategia `add_missing_dependency`, el retest espera a la instalacion controlada:

```text
FIX_APPLIED requirements.txt
  -> DEPENDENCY_INSTALL_REQUESTED
  -> DependencyInstallerAgent ejecuta pip install -r requirements.txt
  -> DEPENDENCY_INSTALL_SUCCEEDED
  -> RETEST_REQUESTED
```

`pip install -r requirements.txt` se ejecuta mediante `CommandTool`, con `shell=False`, `cwd=workspace` y whitelist estricta de argumentos.

`max_fix_attempts` es 2 por defecto y `MAX_EVENTS_PER_WORKFLOW` es 200. Tambien puedes configurarlos mediante `.env` o CLI con `--max-fix-attempts` y `--max-events`.

Los eventos `FIX_REQUESTED`, `FIX_PROPOSED`, `FIX_APPLIED` y `RETEST_REQUESTED` cuentan como progreso real: el supervisor no debe cortar el workflow por limite de eventos mientras haya un fix en curso. Si las correcciones no consiguen que la validacion pase, el workflow termina con `WORKFLOW_HALTED` y `final_failure_reason=max_fix_attempts_exceeded`.

## Failure-aware fixing

`FailureAnalysisService` analiza `stdout` y `stderr` de pytest antes de pedir una correccion. El coordinator crea un `FailureContext` con tipo de fallo, test fallido, linea, traceback, archivos sospechosos, simbolos sospechosos y numero de retry. Ese contexto viaja en la metadata de `FIX_REQUESTED`.

Estrategias actuales:

- `patch_existing_file`
- `add_missing_import`
- `add_missing_dependency`
- `fix_route`
- `fix_test`
- `rewrite_function`
- `strip_markdown_fences`
- `fix_local_module_import`
- `fix_maven_compilation`

`CoderAgent` usa ese contexto para leer archivos relacionados y publicar `FIX_PROPOSED` con `fix_strategy`, `fix_reasoning`, `diff_summary`, `based_on_error` y un hash simple del contenido. `ProjectMemory` guarda errores, fixes aplicados y hashes de fixes para evitar repetir exactamente el mismo cambio.

## Patch Engine

`PatchTool` permite modificar archivos existentes con parches `search/replace` seguros dentro del workspace. No crea archivos nuevos: para archivos nuevos se mantiene `CODE_PROPOSED` y escritura mediante `FileTool`.

Reglas del motor:

- la ruta debe resolverse dentro de `./workspace/`;
- el archivo debe existir y ser textual;
- `search` debe aparecer exactamente una vez;
- si `search` no aparece o aparece varias veces, se publica `PATCH_FAILED`;
- antes de aplicar se conserva un backup interno;
- cada patch aplicado publica `PATCH_APPLIED` con `diff_summary`.

`FileAgent` escucha `PATCH_PROPOSED` y aplica el cambio con `PatchTool`. `ProjectMemory` guarda los patches aplicados y sus resumenes, y `RuntimeSummary` expone `patches_applied` y `patches_failed`.

`fix_local_module_import` corrige fallos como `ModuleNotFoundError: No module named 'app'` alineando `tests/test_app.py` con los simbolos reales exportados por `app.py`: usa `from app import app` si existe `app = Flask(...)`, usa `create_app` solo si existe, y no inventa imports como `db`. Para modulos locales como `app`, `todo` o `models`, el target del fix son archivos Python (`tests/test_app.py`, `app.py`), no `requirements.txt`. Si se intenta aplicar un fix local a `requirements.txt`, `FileAgent` lo rechaza como `wrong_target_fix`.

Para proyectos Spring Boot minimos, el baseline determinista crea un `pom.xml` Spring Boot 3 con Java 17, `DemoApplication`, `HealthController` con `GET /health` y un test `MockMvc` que valida estado 200 y body `OK`.

Para objetivos que contienen `Spring Boot`, `CRUD` y `usuarios`, el baseline crea una API CRUD in-memory con `User`, `UserService`, `UserController` y `UserControllerTest`. El servicio usa `Map<Long, User>`, el controller expone los endpoints `/users` y los tests validan listar, crear, obtener y eliminar usuarios con MockMvc.

Para objetivos Angular minimos, el baseline crea una aplicacion standalone con `package.json`, `angular.json`, `tsconfig.json`, `src/main.ts`, `AppComponent` y template HTML. La validacion controlada publica `BUILD_STARTED`, `BUILD_PASSED` o `BUILD_FAILED` alrededor de `npm run build`.

`FailureAnalysisService` reconoce errores Maven basicos como fallos de compilacion, tests fallidos, dependencias ausentes, version Java incorrecta y paquetes o clases no encontrados.

Limites actuales:

- solo se reemplaza un archivo seguro por fix
- no hay shell libre
- no hay comandos nuevos fuera de la whitelist
- no hay escritura fuera del workspace
- el feedback se basa en `stdout`, `stderr`, exit code y archivos inferidos

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
- `AgentContextBuilder`: objetivo, tarea actual, resumen del grafo, memoria semantica, eventos recientes relevantes, arbol de archivos y archivos del workspace.
- `LLMDecision`: `action`, `reasoning_summary`, `confidence`, `events_to_publish`, `task_updates`, `content`.

Si Ollama no esta disponible o `USE_MOCK_LLM=true`, se usa modo mock. Si el modelo devuelve JSON invalido, el agente cae a una decision determinista segura.

Para Ollama real, el cliente pide `format="json"` y usa opciones conservadoras:

```text
temperature=0
top_p=0.9
num_predict=600
```

Los agentes esperan schemas compactos:

```text
Planner:  {"tasks": [], "reasoning_summary": ""}
Coder:    {"content": "", "reasoning_summary": ""}
Reviewer: {"approved": true, "feedback": "", "reasoning_summary": ""}
```

`JsonExtractionService` intenta parsear JSON directo, quitar fences Markdown, extraer el primer objeto JSON embebido y reparar errores simples como comas finales, booleanos estilo Python o strings con comillas simples.

## Troubleshooting Ollama

Modelos recomendados para este proyecto:

- `llama3.2`
- `qwen2.5-coder`
- `mistral`

Si ves fallbacks altos, normalmente significa que Ollama devolvio texto no parseable, el modelo no existe localmente o el endpoint respondio con error. El sistema no ejecuta comandos ni se detiene por eso: registra el fallo, usa una decision determinista segura y continua cuando puede.

Las trazas LLM se guardan en:

```text
workspace/.traces/
```

Cada trace contiene prompt resumido, respuesta cruda, respuesta parseada, razon del fallback si aplica y latencia. La carpeta `.traces` no se incluye como archivo creado por el objetivo.

## Memoria de proyecto

`ProjectMemoryService` mantiene una memoria por `correlation_id` y la persiste en SQLite. Se actualiza automaticamente cuando el bus publica eventos relevantes como `GOAL_DECOMPOSED`, `CODE_PROPOSED`, `FILE_WRITTEN`, `PROJECT_REVIEW_APPROVED`, `PROJECT_REVIEW_REJECTED`, `TEST_PASSED` y `TEST_FAILED`.

La memoria resume:

- objetivo del proyecto
- framework detectado
- archivos creados
- decisiones de arquitectura
- convenciones de codigo
- errores conocidos
- feedback del reviewer
- tareas completadas relevantes

El contexto enviado al LLM queda limitado por `AgentContextBuilder`: se prioriza la memoria resumida, el arbol de archivos y los eventos recientes de mayor senal, evitando reenviar todo el historial completo.

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
