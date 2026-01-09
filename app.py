from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, send_from_directory, flash
import os, json, uuid, subprocess, threading, time, re, tempfile, shutil, csv, hashlib
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
from collections import OrderedDict

app = Flask(__name__, static_folder='.')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def wants_json_response():
    """Return True if this request expects a JSON response (AJAX/API)."""
    if request.is_json:
        return True
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return True
    accept = (request.headers.get('Accept') or '').lower()
    return 'application/json' in accept


# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('exports', exist_ok=True)
os.makedirs('backups', exist_ok=True)
os.makedirs('img', exist_ok=True)  # Ensure img directory exists

# Configure logging
# Logging (structured-ish): include execution_id/tool_id when provided via 'extra'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_log_fmt = "%(asctime)s %(levelname)s %(name)s exec=%(execution_id)s tool=%(tool_id)s - %(message)s"
class _SafeExtraFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "execution_id"):
            record.execution_id = "-"
        if not hasattr(record, "tool_id"):
            record.tool_id = "-"
        return super().format(record)

_formatter = _SafeExtraFormatter(_log_fmt)

# Console handler
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_formatter)
logger.addHandler(_ch)

# File handler (rotating)
try:
    from logging.handlers import RotatingFileHandler
    _fh = RotatingFileHandler("logs/app.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_formatter)
    logger.addHandler(_fh)
except Exception as _e:
    logger.warning(f"Failed to initialize file logging: {_e}")


# Database simulation
workflows_db = OrderedDict()
execution_history = OrderedDict()
update_history = OrderedDict()

# Execution control (in-memory)
# - active_processes: current subprocess (per execution) so we can terminate on cancel
# - cancel_events: cancellation signal (per execution)
active_processes: Dict[str, subprocess.Popen] = {}
cancel_events: Dict[str, threading.Event] = {}

# Concurrency control for shared in-memory state
execution_lock = threading.RLock()

def _sha256_text(value: str) -> str:
    try:
        return hashlib.sha256((value or "").encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return ""

def _now_iso() -> str:
    return datetime.now().isoformat()

def _int_or_none(v) -> Optional[int]:
    """Best-effort int parsing for values coming from JSON/forms."""
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip() == "":
            return None
        return int(v)
    except Exception:
        return None

def _record_event(execution, event_type: str, message: str, level: str = "info", tool_id: str = None):
    """Append a structured event to an execution and bump its version."""
    evt = {
        "ts": _now_iso(),
        "type": event_type,
        "level": level,
        "tool_id": tool_id,
        "message": message
    }
    with execution_lock:
        if not hasattr(execution, "events") or execution.events is None:
            execution.events = []
        execution.events.append(evt)
        # avoid unbounded growth in memory (keep latest 500)
        if len(execution.events) > 500:
            execution.events = execution.events[-500:]
        execution.version = int(getattr(execution, "version", 0)) + 1
        execution.last_updated_at = _now_iso()

# Settings file path
SETTINGS_FILE = 'app_settings.json'

# Default settings - only essential workflow tool settings
DEFAULT_SETTINGS = {
    'max_execution_time': 9999,        # Maximum time per tool in seconds
    'max_output_size': 999999,         # Maximum output size in characters
    'auto_cleanup': True,             # Automatically clean up temp files
    'default_theme': 'dark',          # UI theme
    'enable_logging': True,           # Enable debug logging
    'concurrent_executions': 3,       # Max concurrent workflow executions
    'log_retention_days': 7,         # Days to keep logs
    'backup_frequency': 'weekly',     # Auto-backup frequency
    'enable_notifications': False,    # Enable desktop notifications
    'enable_auto_updates': False,      # Check for tool updates
    'output_directory': 'exports',    # Default output directory
    'temp_directory': '',             # Custom temp directory (empty = system default)
    'enable_debug_mode': False,       # Enable debug mode for tools
    'tool_library': [],              # Global configured tools
    'wordlists': {}                  # Named wordlists e.g. {'wordlist1':'/path/to/file'}
}

class ToolConfig:
    """Configuration for a single tool/module"""
    # input_source historically supported: 'initial', 'previous', 'specific', 'none'.
    # The workflow editor now defaults to 'specific' ("any tool") and hides the legacy options,
    # but we keep backward-compatibility for previously saved workflows.
    def __init__(self, tool_id, name, command='', args_template='', input_source='specific',
                 library_tool_id=None,
                 command_override=None,
                 args_template_override=None,
                 update_command_override=None,
                 description_override=None,
                 specific_step=None, description="", color="#4a86e8", enabled=True,
                 update_command="", last_updated=None, 
                 input_method='argument',
                 output_handling='stdout',
                 provides_output=True,
                 output_format='text',
                 output_file_path='',
                 placeholder_name='input'):
        self.id = tool_id
        self.name = name
        self.library_tool_id = library_tool_id
        # Overrides: if None, value will be resolved from tool library at execution time
        self.command_override = command_override
        self.args_template_override = args_template_override
        self.update_command_override = update_command_override
        self.description_override = description_override

        # Backward-compatible fields (may be empty when using tool library)
        self.command = command
        self.args_template = args_template
        self.input_source = input_source
        self.specific_step = specific_step
        self.description = description
        self.color = color
        self.enabled = enabled
        self.update_command = update_command or ""
        self.last_updated = last_updated
        # New attributes for enhanced workflow handling
        self.input_method = input_method
        self.output_handling = output_handling
        self.provides_output = provides_output
        self.output_format = output_format
        self.output_file_path = output_file_path
        self.placeholder_name = placeholder_name
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'command': self.command,
            'args_template': self.args_template,
            'library_tool_id': self.library_tool_id,
            'command_override': self.command_override,
            'args_template_override': self.args_template_override,
            'update_command_override': self.update_command_override,
            'description_override': self.description_override,
            'input_source': self.input_source,
            'specific_step': self.specific_step,
            'description': self.description,
            'color': self.color,
            'enabled': self.enabled,
            'update_command': self.update_command,
            'last_updated': self.last_updated,
            'input_method': self.input_method,
            'output_handling': self.output_handling,
            'provides_output': self.provides_output,
            'output_format': self.output_format,
            'output_file_path': self.output_file_path,
            'placeholder_name': self.placeholder_name
        }

    @classmethod
    def from_dict(cls, data):
        """Create ToolConfig from dictionary with backward compatibility"""
        return cls(
            tool_id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            command=data.get('command',''),
            args_template=data.get('args_template',''),
            input_source=data.get('input_source', 'specific'),
            library_tool_id=data.get('library_tool_id'),
            command_override=data.get('command_override'),
            args_template_override=data.get('args_template_override'),
            update_command_override=data.get('update_command_override'),
            description_override=data.get('description_override'),
            specific_step=data.get('specific_step'),
            description=data.get('description', ''),
            color=data.get('color', '#4a86e8'),
            enabled=data.get('enabled', True),
            update_command=data.get('update_command', ''),
            last_updated=data.get('last_updated'),
            input_method=data.get('input_method', 'argument'),
            output_handling=data.get('output_handling', 'stdout'),
            provides_output=data.get('provides_output', True),
            output_format=data.get('output_format', 'text'),
            output_file_path=data.get('output_file_path', ''),
            placeholder_name=data.get('placeholder_name', 'input')
        )

class Workflow:
    """Complete workflow configuration"""
    def __init__(
        self,
        workflow_id,
        name,
        description,
        tools,
        created_at,
        updated_at,
        author="anonymous",
        run_mode: str = "once",
        interval_minutes: Optional[int] = None,
        repeat_count: Optional[int] = None,
        repeat_interval_minutes: Optional[int] = None,
    ):
        self.id = workflow_id
        self.name = name
        self.description = description
        self.tools = tools
        self.created_at = created_at
        self.updated_at = updated_at
        self.author = author
        # Recurrence configuration
        # - run_mode: 'once' | 'interval' | 'repeat'
        # - interval: run indefinitely every N minutes
        # - repeat: run repeat_count times, sleeping repeat_interval_minutes between runs
        self.run_mode = (run_mode or "once").strip().lower()
        self.interval_minutes = interval_minutes
        self.repeat_count = repeat_count
        self.repeat_interval_minutes = repeat_interval_minutes
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tools': [tool.to_dict() for tool in self.tools],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'author': self.author,
            'run_mode': getattr(self, 'run_mode', 'once'),
            'interval_minutes': getattr(self, 'interval_minutes', None),
            'repeat_count': getattr(self, 'repeat_count', None),
            'repeat_interval_minutes': getattr(self, 'repeat_interval_minutes', None),
        }

class ExecutionResult:
    """Result of a tool execution"""
    def __init__(self, tool_id, tool_name, status='pending', start_time=None, 
                 end_time=None, output="", error="", exit_code=None):
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.output = output
        self.error = error
        self.exit_code = exit_code
    
    def to_dict(self):
        return {
            'tool_id': self.tool_id,
            'tool_name': self.tool_name,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'output': self.output,
            'error': self.error,
            'exit_code': self.exit_code
        }

class WorkflowExecution:
    """Complete execution instance"""
    def __init__(self, execution_id, workflow_id, domain, status='queued',
                 results=None, created_at=None, started_at=None, completed_at=None,
                 version: int = 0, last_updated_at: str = None, events: list = None,
                 cancel_requested: bool = False, cancelled_at: str = None,
                 run_mode: str = 'once',
                 interval_minutes: Optional[int] = None,
                 repeat_count: Optional[int] = None,
                 repeat_interval_minutes: Optional[int] = None,
                 current_iteration: int = 0,
                 planned_iterations: Optional[int] = None,
                 iterations: Optional[list] = None):
        self.execution_id = execution_id
        self.workflow_id = workflow_id
        self.domain = domain
        self.status = status
        self.results = results or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.started_at = started_at
        self.completed_at = completed_at
        self.version = int(version or 0)
        self.last_updated_at = last_updated_at or self.created_at
        self.events = events or []
        self.cancel_requested = bool(cancel_requested)
        self.cancelled_at = cancelled_at
        # Recurrence metadata (non-breaking: kept optional and ignored by older UIs)
        self.run_mode = (run_mode or 'once')
        self.interval_minutes = interval_minutes
        self.repeat_count = repeat_count
        self.repeat_interval_minutes = repeat_interval_minutes
        self.current_iteration = int(current_iteration or 0)
        self.planned_iterations = planned_iterations
        # iterations is a list of per-iteration summaries + results snapshots
        self.iterations = iterations or []

    def to_dict(self, include_results: bool = True):
        data = {
            'execution_id': self.execution_id,
            'workflow_id': self.workflow_id,
            'domain': self.domain,
            'status': self.status,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'version': self.version,
            'last_updated_at': self.last_updated_at,
            'events': self.events,
            'cancel_requested': getattr(self, 'cancel_requested', False),
            'cancelled_at': getattr(self, 'cancelled_at', None)
            ,
            'run_mode': getattr(self, 'run_mode', 'once'),
            'interval_minutes': getattr(self, 'interval_minutes', None),
            'repeat_count': getattr(self, 'repeat_count', None),
            'repeat_interval_minutes': getattr(self, 'repeat_interval_minutes', None),
            'current_iteration': getattr(self, 'current_iteration', 0),
            'planned_iterations': getattr(self, 'planned_iterations', None),
            'iterations': getattr(self, 'iterations', []),
        }
        if include_results:
            # Snapshot under lock to avoid races while the execution thread is updating results.
            with execution_lock:
                snapshot = dict(self.results or {})
            data['results'] = {k: v.to_dict() for k, v in snapshot.items()}
        return data

class UpdateResult:
    """Result of a tool update"""
    def __init__(self, update_id, tool_id, tool_name, status='pending', 
                 start_time=None, end_time=None, output="", error=""):
        self.update_id = update_id
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.output = output
        self.error = error
    
    def to_dict(self):
        return {
            'update_id': self.update_id,
            'tool_id': self.tool_id,
            'tool_name': self.tool_name,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'output': self.output,
            'error': self.error
        }

# Helper Functions
def parse_args_template(template: str, context: Dict[str, Any]) -> str:
    """Parse argument template with placeholders like {0}, {1}, {domain}, {input}, etc."""
    logger.debug(f"Parsing template: {template}")
    logger.debug(f"Context keys: {list(context.keys())}")
    
    for key, value in context.items():
        placeholder = f"{{{key}}}"
        if placeholder in template:
            # If value is a file path and contains spaces, quote it.
            # Only apply this to placeholders that are likely to be file paths to avoid
            # accidentally quoting arbitrary strings that happen to match an existing path.
            if isinstance(value, str) and os.path.exists(value) and ' ' in value:
                value = f'"{value}"'
            logger.debug(f"Replacing {placeholder} with: {value[:100] if isinstance(value, str) and len(value) > 100 else value}")
            template = template.replace(placeholder, str(value))
    
    logger.debug(f"Parsed template result: {template}")
    return template

def resolve_tool_from_library(tool: ToolConfig, settings: Dict[str, Any]) -> Dict[str, str]:
    """Resolve effective command/args/update/description for a workflow tool step.

    If tool.library_tool_id is set, merge the library tool definition with any per-step overrides.
    """
    library = settings.get('tool_library', []) or []
    lib_entry = None
    if tool.library_tool_id:
        for t in library:
            if t.get('id') == tool.library_tool_id:
                lib_entry = t
                break

    def pick(field_name: str, override_value, fallback_value: str) -> str:
        if override_value is not None and str(override_value).strip() != "":
            return str(override_value)
        if fallback_value is None:
            return ""
        return str(fallback_value)

    if lib_entry:
        effective_command = pick('path', tool.command_override, lib_entry.get('path', lib_entry.get('command', '')))
        effective_args = pick('default_command', tool.args_template_override, lib_entry.get('default_command', lib_entry.get('args_template', '')))
        effective_update = pick('update_command', tool.update_command_override, lib_entry.get('update_command', ''))
        effective_desc = pick('description', tool.description_override, lib_entry.get('description', ''))
    else:
        # No library reference; use the legacy per-step fields
        effective_command = pick('command', tool.command_override, tool.command)
        effective_args = pick('args_template', tool.args_template_override, tool.args_template)
        effective_update = pick('update_command', tool.update_command_override, tool.update_command)
        effective_desc = pick('description', tool.description_override, tool.description)

    return {
        'command': effective_command,
        'args_template': effective_args,
        'update_command': effective_update,
        'description': effective_desc,
    }
    return template

def find_tool_by_id(workflow, tool_id):
    """Find tool in workflow by ID"""
    for tool in workflow.tools:
        if tool.id == tool_id:
            return tool
    return None

def find_tool_id_by_step(workflow, step_number):
    """Find tool ID by step number (1-indexed)"""
    if 1 <= step_number <= len(workflow.tools):
        return workflow.tools[step_number - 1].id
    return None

def _terminate_process_tree(proc: subprocess.Popen):
    """Best-effort termination of a running subprocess (and its process group)."""
    if proc is None:
        return
    try:
        if os.name == 'nt':
            # On Windows, terminate the process; child processes may require additional handling.
            proc.terminate()
        else:
            # On POSIX, kill the whole process group (if created).
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
    except Exception:
        pass


def _run_command_cancellable(execution_id: str, cmd: str, stdin_data: Optional[str], timeout_seconds: int, cancel_event: Optional[threading.Event]):
    """Run a shell command with cancellation support.

    Returns: (returncode, stdout, stderr)
    Raises: subprocess.TimeoutExpired
    """
    start = time.time()

    # Configure process group so we can terminate the whole tree on POSIX.
    popen_kwargs = {
        "shell": True,
        "stdin": subprocess.PIPE if stdin_data is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }

    if os.name == 'nt':
        # CREATE_NEW_PROCESS_GROUP enables sending ctrl events, but terminate() is enough for now.
        try:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
        import os as _os
        popen_kwargs["preexec_fn"] = _os.setsid

    proc = subprocess.Popen(cmd, **popen_kwargs)
    with execution_lock:
        active_processes[execution_id] = proc

    try:
        # Poll loop so we can react to cancellation.
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process_tree(proc)
                break

            if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
                _terminate_process_tree(proc)
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.1)

        try:
            stdout, stderr = proc.communicate(input=stdin_data, timeout=5)
        except Exception:
            # Best-effort: if the process is still alive, force-terminate and re-collect.
            try:
                _terminate_process_tree(proc)
            except Exception:
                pass
            stdout, stderr = proc.communicate(input=stdin_data)
        return proc.returncode, stdout or "", stderr or ""
    finally:
        with execution_lock:
            # Only clear if we're still the current process for this execution.
            if active_processes.get(execution_id) is proc:
                active_processes.pop(execution_id, None)


def execute_tool(tool: ToolConfig, domain: str, previous_outputs: Dict[str, Dict[str, Any]],
                 temp_dir: str, execution_context: Dict[str, Any],
                 cancel_event: Optional[threading.Event] = None,
                 execution_id: Optional[str] = None) -> ExecutionResult:
    """Execute a single tool with enhanced input/output handling"""
    logger.info(f"=== STARTING EXECUTION OF TOOL: {tool.name} (ID: {tool.id}) ===")
    settings = load_settings()
    resolved = resolve_tool_from_library(tool, settings)
    logger.info(f"Tool configuration (resolved):")
    logger.info(f"  - Command/Path: {resolved['command']}")
    logger.info(f"  - Default Command (args): {resolved['args_template']}")
    logger.info(f"  - Input source: {tool.input_source}")
    logger.info(f"  - Input method: {tool.input_method}")
    logger.info(f"  - Placeholder name: {tool.placeholder_name}")
    logger.info(f"  - Output handling: {tool.output_handling}")
    logger.info(f"  - Provides output: {tool.provides_output}")
    
    result = ExecutionResult(
        tool_id=tool.id,
        tool_name=tool.name,
        status='running',
        start_time=datetime.now().isoformat()
    )
    
    try:
        # Build execution context with dynamic placeholders
        context = {
            'domain': domain,
            'temp_dir': temp_dir,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'tool_id': tool.id
        }
        
        # Add domain as {0} for backward compatibility
        context['0'] = domain
        
        logger.info(f"Initial context keys: {list(context.keys())}")
        logger.info(f"Previous outputs available from tools: {list(previous_outputs.keys())}")
        
        # Store previous outputs in the template context.
        # Notes:
        # - We keep the existing behavior of exposing outputs via placeholder_name and
        #   indexed placeholders for backward compatibility.
        # - To avoid silent collisions (multiple tools using the same placeholder_name),
        #   we also expose stable per-tool keys: out_<tool_id> and step_<n>.
        for idx, (tool_id, output_data) in enumerate(previous_outputs.items()):
            # Get the tool that produced this output
            prev_tool = find_tool_by_id(execution_context['workflow'], tool_id)
            if prev_tool:
                # Use the previous tool's placeholder name
                placeholder = prev_tool.placeholder_name
                # Preserve any previous value if this placeholder is reused.
                if placeholder in context and context.get(placeholder) != output_data['output']:
                    context[f"{placeholder}_{idx}"] = context.get(placeholder)
                context[placeholder] = output_data['output']
                # Also add as indexed placeholder for backward compatibility
                context[str(idx + 1)] = output_data['output']
                # Stable keys (recommended)
                context[f"out_{tool_id}"] = output_data['output']
                context[f"step_{idx + 1}"] = output_data['output']
                logger.info(f"Added output from tool {prev_tool.name} (ID: {tool_id}) to context:")
                logger.info(f"  - Key '{placeholder}': {output_data['output'][:200]}...")
                logger.info(f"  - Also available as '{idx + 1}'")
                logger.info(f"  - Also available as 'out_{tool_id}' and 'step_{idx + 1}'")
        
        # Handle input based on input_method
        input_content = None
        logger.info(f"Determining input for tool {tool.name}:")
        logger.info(f"  - Input source setting: {tool.input_source}")
        logger.info(f"  - Specific step (if set): {tool.specific_step}")
        
        if tool.input_source != 'none' and tool.input_method != 'none':
            if tool.input_source == 'specific':
                # Use a specific step's output ("any tool").
                # If the configured step is missing/invalid, fall back safely.
                step = None
                try:
                    step = int(tool.specific_step) if tool.specific_step is not None else None
                except Exception:
                    step = None

                if step:
                    source_tool_id = find_tool_id_by_step(execution_context['workflow'], step)
                    logger.info(f"  Looking for specific step {step} -> tool ID: {source_tool_id}")
                    if source_tool_id and source_tool_id in previous_outputs:
                        input_content = previous_outputs[source_tool_id]['output']
                        logger.info(f"  Found input from specific tool {source_tool_id}")
                    else:
                        logger.warning(f"  Could not find output from specific step {step}; will fall back")

                # Fallback if step not set or not found
                if input_content is None:
                    if previous_outputs:
                        last_tool_id = list(previous_outputs.keys())[-1]
                        input_content = previous_outputs[last_tool_id]['output']
                        logger.info(f"  Fallback to most recent output (ID: {last_tool_id})")
                    else:
                        input_content = domain
                        logger.info(f"  Fallback to domain as input: {domain}")

            elif tool.input_source == 'initial':
                input_content = domain
                logger.info(f"  Using initial domain as input: {domain}")
            else:  # 'previous' or default
                # Use the most recent output
                if previous_outputs:
                    last_tool_id = list(previous_outputs.keys())[-1]
                    input_content = previous_outputs[last_tool_id]['output']
                    logger.info(f"  Using output from previous tool (ID: {last_tool_id})")
                    logger.info(f"  Input content size: {len(input_content)} characters")
                else:
                    logger.warning(f"  No previous outputs available, will use domain as fallback")
                    input_content = domain
        
        # Prepare input based on input_method
        stdin_data = None
        temp_files_to_cleanup = []
        
        if input_content and tool.input_method != 'none':
            if tool.input_method == 'file':
                # Create temp file with input content
                logger.info(f"Creating temporary file for tool input (input_method='file')")
                fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix='.txt')
                temp_files_to_cleanup.append(temp_path)
                with os.fdopen(fd, 'w') as f:
                    f.write(input_content)
                # Add the file path to context for the template
                context[tool.placeholder_name] = temp_path
                context['input_file'] = temp_path
                logger.info(f"Created temp file: {temp_path}")
                logger.info(f"File content size: {len(input_content)} characters")
            elif tool.input_method == 'stdin':
                # Store for stdin input during execution
                stdin_data = input_content
                # For template, use '-' to indicate stdin
                context[tool.placeholder_name] = '-'
                logger.info(f"Prepared stdin input (size: {len(input_content)} chars)")
            elif tool.input_method == 'argument':
                # Check if content is multiline - if so, we need to create a temp file
                # because command-line arguments can't handle multiline strings properly
                if '\n' in input_content and len(input_content.strip().split('\n')) > 1:
                    logger.warning(f"Multiline content detected for argument input. Creating temp file instead.")
                    logger.info(f"Content has {len(input_content.strip().splitlines())} lines")
                    
                    # Create temp file
                    fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix='.txt')
                    temp_files_to_cleanup.append(temp_path)
                    with os.fdopen(fd, 'w') as f:
                        f.write(input_content)
                    
                    # Replace the placeholder in args_template that needs the file
                    # We need to find which placeholder in the args_template is being used
                    args_template = tool.args_template
                    
                    # Check if the template has file-related flags
                    if any(flag in args_template for flag in ['-l', '-i', '--input', '-f', '@']):
                        logger.info(f"Template has file flag, replacing placeholder with file path")
                        # Replace the appropriate placeholder with file path
                        # We'll replace both the named placeholder and indexed placeholder
                        if '{' + tool.placeholder_name + '}' in args_template:
                            context[tool.placeholder_name] = temp_path
                            logger.info(f"Replaced placeholder '{tool.placeholder_name}' with file path")
                        
                        # Also replace indexed placeholders that might be referencing this input
                        for key in list(context.keys()):
                            if key.isdigit() and context[key] == input_content:
                                context[key] = temp_path
                                logger.info(f"Replaced indexed placeholder '{key}' with file path")
                    else:
                        # For non-file arguments, we need to handle multiline content
                        # Join lines with spaces or handle differently based on tool
                        input_content = ' '.join(input_content.strip().split('\n'))
                        context[tool.placeholder_name] = input_content
                        logger.info(f"Joined multiline content into single line argument")
                else:
                    # Single line content, safe to pass as argument
                    context[tool.placeholder_name] = input_content
                    logger.info(f"Setting placeholder '{tool.placeholder_name}' as single-line argument")
        
        # Prepare output file if needed
        if tool.output_handling == 'file':
            if tool.output_file_path:
                output_file = tool.output_file_path
            else:
                output_file = os.path.join(temp_dir, f"output_{tool.id}_{context['timestamp']}.txt")
            context['output_file'] = output_file
            logger.info(f"Output will be written to file: {output_file}")
        
        logger.info(f"Final context before parsing template:")
        for key, value in context.items():
            if key in ['input_file', 'output_file'] or key == tool.placeholder_name:
                logger.info(f"  {key}: {value}")
            elif isinstance(value, str) and len(value) > 100:
                logger.info(f"  {key}: {value[:100]}...")
            else:
                logger.info(f"  {key}: {value}")
        
        # Parse arguments with context
        # Inject global wordlists into context (usable as {wordlistName})
        for wl_name, wl_path in (settings.get('wordlists', {}) or {}).items():
            context[wl_name] = wl_path

        args = parse_args_template(resolved['args_template'], context)

        # Build command
        cmd = f"{resolved['command']} {args}".strip()
        logger.info(f"FINAL COMMAND TO EXECUTE: {cmd}")
        
        # Get timeout from settings
        settings = load_settings()
        timeout_value = int(settings.get('max_execution_time', 300))
        
        # Execute with appropriate input method (cancellable)
        logger.info(f"Executing command with timeout {timeout_value} seconds...")

        exec_id = execution_id or execution_context.get('execution_id')
        if not exec_id:
            # Fallback: no execution id; run without cancellability bookkeeping.
            process = subprocess.run(
                cmd,
                shell=True,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout_value
            )
            rc, stdout, stderr = process.returncode, process.stdout, process.stderr
        else:
            rc, stdout, stderr = _run_command_cancellable(
                execution_id=str(exec_id),
                cmd=cmd,
                stdin_data=stdin_data,
                timeout_seconds=timeout_value,
                cancel_event=cancel_event
            )

        result.exit_code = rc
        
        # If output is to a file, read it
        if tool.output_handling == 'file' and 'output_file' in context and os.path.exists(context['output_file']):
            with open(context['output_file'], 'r') as f:
                result.output = f.read()
            logger.info(f"Read output from file: {context['output_file']}")
        else:
            result.output = stdout
        
        result.error = stderr
        # If the execution was cancelled mid-tool, mark status accordingly.
        if cancel_event is not None and cancel_event.is_set():
            result.status = 'cancelled'
        else:
            result.status = 'completed' if rc == 0 else 'failed'
        
        logger.info(f"Command execution completed:")
        logger.info(f"  - Exit code: {rc}")
        logger.info(f"  - Status: {result.status}")
        logger.info(f"  - Output size: {len(result.output)} characters")
        logger.info(f"  - Error size: {len(result.error)} characters")
        
        if rc != 0 and result.status != 'cancelled':
            logger.error(f"Command failed with exit code {rc}")
            logger.error(f"STDOUT (first 500 chars): {result.output[:500]}")
            logger.error(f"STDERR (first 500 chars): {result.error[:500]}")
        else:
            if result.status == 'cancelled':
                logger.warning("Command was cancelled by the user")
            else:
                logger.info(f"Command succeeded")
                logger.debug(f"Output (first 500 chars): {result.output[:500]}")
        
    except subprocess.TimeoutExpired:
        settings = load_settings()
        timeout_value = int(settings.get('max_execution_time', 300))
        result.status = 'failed'
        result.error = f"Command timed out after {timeout_value} seconds"
        logger.error(f"Command timed out after {timeout_value} seconds")
    except Exception as e:
        result.status = 'failed'
        result.error = str(e)
        logger.error(f"Exception during execution: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Clean up temp files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    result.end_time = datetime.now().isoformat()
    logger.info(f"=== COMPLETED EXECUTION OF TOOL: {tool.name} ===")
    return result

def execute_update_thread(tool: ToolConfig, workflow_id: str):
    """Background thread to execute tool update"""
    update_id = str(uuid.uuid4())
    result = UpdateResult(
        update_id=update_id,
        tool_id=tool.id,
        tool_name=tool.name,
        status='running',
        start_time=datetime.now().isoformat()
    )
    
    update_history[update_id] = result
    
    try:
        if not tool.update_command:
            result.status = 'failed'
            result.error = "No update command configured"
            return
        
        # Execute the update command.
        # Convention: update_command is an argument string appended to the tool executable.
        # (Kept as shell=True for backward compatibility with existing update commands.)
        cmd = f"{tool.command} {tool.update_command}".strip()
        logger.info(f"Updating tool: {cmd}")
        
        # Get timeout from settings
        settings = load_settings()
        timeout_value = int(settings.get('max_execution_time', 300))
        
        # Run the command with a timeout
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_value
        )
        
        result.output = process.stdout
        result.error = process.stderr
        result.status = 'completed' if process.returncode == 0 else 'failed'
        
        # Update the tool's last_updated timestamp
        if result.status == 'completed':
            tool.last_updated = datetime.now().isoformat()

            # If this is a library tool update, persist last_updated back into settings.
            if workflow_id == '__library__':
                settings = load_settings()
                lib = settings.get('tool_library', []) or []
                for entry in lib:
                    if entry.get('id') == tool.id:
                        entry['last_updated'] = tool.last_updated
                        break
                save_settings(settings)
            else:
                # Save the workflow with updated timestamp
                workflow = workflows_db.get(workflow_id)
                if workflow:
                    # Find and update the tool in the workflow
                    for wf_tool in workflow.tools:
                        if wf_tool.id == tool.id:
                            wf_tool.last_updated = tool.last_updated
                    workflow.updated_at = datetime.now().isoformat()
        
    except subprocess.TimeoutExpired:
        settings = load_settings()
        timeout_value = int(settings.get('max_execution_time', 300))
        result.status = 'failed'
        result.error = f"Update timed out after {timeout_value} seconds"
    except Exception as e:
        result.status = 'failed'
        result.error = str(e)
    
    result.end_time = datetime.now().isoformat()
    return result

def run_workflow_thread(execution: WorkflowExecution):
    """Background thread to execute workflow with enhanced data flow"""
    app_settings = load_settings()
    base_temp = (app_settings.get("temp_directory") or "").strip()
    if base_temp:
        os.makedirs(base_temp, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="workflow_", dir=base_temp)
    else:
        temp_dir = tempfile.mkdtemp(prefix="workflow_")
    previous_outputs = {}  # Store outputs by tool ID

    # Cancellation signal for this execution (created when the execution is enqueued).
    cancel_event = None
    with execution_lock:
        cancel_event = cancel_events.get(execution.execution_id)
    
    try:
        workflow = workflows_db.get(execution.workflow_id)
        if not workflow:
            with execution_lock:
                execution.status = 'failed'
            return

        with execution_lock:
            execution.status = 'running'
            execution.started_at = datetime.now().isoformat()
        _record_event(execution, 'execution_started', f"Execution started for domain: {execution.domain}")
        logger.info(f"=== STARTING WORKFLOW EXECUTION ===")
        logger.info(f"Execution ID: {execution.execution_id}")
        logger.info(f"Workflow ID: {execution.workflow_id}")
        logger.info(f"Domain: {execution.domain}")
        logger.info(f"Number of tools in workflow: {len(workflow.tools)}")
        
        # Log workflow tools configuration
        for i, tool in enumerate(workflow.tools):
            logger.info(f"Tool {i+1}: {tool.name}")
            logger.info(f"  - ID: {tool.id}")
            logger.info(f"  - Command: {tool.command}")
            logger.info(f"  - Args: {tool.args_template}")
            logger.info(f"  - Input source: {tool.input_source}")
            logger.info(f"  - Input method: {tool.input_method}")
            logger.info(f"  - Placeholder: {tool.placeholder_name}")
            logger.info(f"  - Enabled: {tool.enabled}")
        
        # Create execution context to pass to tools
        execution_context = {
            'workflow': workflow,
            'execution_id': execution.execution_id,
            'temp_dir': temp_dir
        }
        
        # Determine recurrence plan
        mode = (getattr(workflow, 'run_mode', 'once') or 'once').strip().lower()
        interval_minutes = _int_or_none(getattr(workflow, 'interval_minutes', None))
        repeat_count = _int_or_none(getattr(workflow, 'repeat_count', None))
        repeat_interval_minutes = _int_or_none(getattr(workflow, 'repeat_interval_minutes', None))

        # Normalize invalid configs to safe defaults
        if mode not in ('once', 'interval', 'repeat'):
            mode = 'once'
        if mode == 'interval' and (not interval_minutes or interval_minutes <= 0):
            logger.warning("Invalid interval_minutes for recurring workflow; falling back to single run")
            mode = 'once'
        if mode == 'repeat':
            if not repeat_count or repeat_count <= 0:
                logger.warning("Invalid repeat_count for recurring workflow; falling back to single run")
                mode = 'once'
            if not repeat_interval_minutes or repeat_interval_minutes < 0:
                repeat_interval_minutes = 0

        planned_iterations = None
        if mode == 'repeat':
            planned_iterations = repeat_count

        with execution_lock:
            execution.run_mode = mode
            execution.interval_minutes = interval_minutes
            execution.repeat_count = repeat_count
            execution.repeat_interval_minutes = repeat_interval_minutes
            execution.planned_iterations = planned_iterations
            execution.current_iteration = 0

        def _sleep_cancellable(seconds: int) -> bool:
            """Sleep up to 'seconds' seconds. Returns False if cancelled."""
            if seconds <= 0:
                return True
            if cancel_event is None:
                time.sleep(seconds)
                return True
            # Wait in a single call so cancellation wakes us immediately.
            return not cancel_event.wait(timeout=seconds)

        iteration = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                with execution_lock:
                    execution.status = 'cancelled'
                    execution.cancel_requested = True
                    execution.cancelled_at = datetime.now().isoformat()
                _record_event(execution, 'execution_cancelled', 'Execution cancelled by user')
                logger.warning("Execution cancelled before starting next iteration")
                break

            iteration += 1
            with execution_lock:
                execution.current_iteration = iteration
                execution.last_updated_at = _now_iso()

            iter_started_at = datetime.now().isoformat()
            _record_event(execution, 'iteration_started', f"Iteration {iteration} started", level='info')
            logger.info(f"\n{'#'*72}")
            logger.info(f"STARTING ITERATION {iteration} (mode={mode})")
            logger.info(f"{'#'*72}")

            # Each iteration starts with a fresh data-flow state.
            previous_outputs = {}
            iter_results_snapshot = {}

            # Execute tools in order (single iteration)
            for idx, tool in enumerate(workflow.tools):
                if cancel_event is not None and cancel_event.is_set():
                    with execution_lock:
                        execution.status = 'cancelled'
                        execution.cancel_requested = True
                        execution.cancelled_at = datetime.now().isoformat()
                    _record_event(execution, 'execution_cancelled', 'Execution cancelled by user')
                    logger.warning("Execution cancelled before starting next tool")
                    break

                if not tool.enabled:
                    logger.info(f"Skipping disabled tool: {tool.name}")
                    _record_event(execution, 'tool_skipped', f"Tool skipped (disabled): {tool.name}", tool_id=tool.id)
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"EXECUTING TOOL {idx+1}/{len(workflow.tools)}: {tool.name}")
                logger.info(f"{'='*60}")

                _record_event(execution, 'tool_started', f"Tool started: {tool.name}", tool_id=tool.id)
                result = execute_tool(
                    tool,
                    execution.domain,
                    previous_outputs,
                    temp_dir,
                    execution_context,
                    cancel_event=cancel_event,
                    execution_id=execution.execution_id
                )

                with execution_lock:
                    # Keep latest results per tool id for compatibility with existing UI.
                    execution.results[tool.id] = result

                iter_results_snapshot[tool.id] = result.to_dict()
                _record_event(
                    execution,
                    'tool_finished',
                    f"Tool finished: {tool.name} ({result.status})",
                    level=('error' if result.status == 'failed' else 'info'),
                    tool_id=tool.id
                )

                if result.status == 'completed' and tool.provides_output:
                    previous_outputs[tool.id] = {
                        'output': result.output,
                        'tool_name': tool.name,
                        'output_format': tool.output_format
                    }
                    logger.info(f"✓ Tool {tool.name} completed successfully, output stored")
                    logger.info(f"  Output size: {len(result.output)} characters")
                    logger.info(f"  Output preview: {result.output[:200]}...")
                else:
                    logger.warning(f"✗ Tool {tool.name} failed or doesn't provide output: {result.status}")
                    if result.error:
                        logger.warning(f"  Error: {result.error[:500]}")

                if result.status == 'cancelled':
                    with execution_lock:
                        execution.status = 'cancelled'
                        execution.cancel_requested = True
                        execution.cancelled_at = datetime.now().isoformat()
                    _record_event(execution, 'execution_cancelled', 'Execution cancelled by user')
                    logger.warning("Execution cancelled during tool execution")
                    break

            # Snapshot iteration results and metadata
            iter_completed_at = datetime.now().isoformat()
            iter_status = 'cancelled' if (cancel_event is not None and cancel_event.is_set()) else 'completed'
            with execution_lock:
                execution.iterations.append({
                    'iteration': iteration,
                    'started_at': iter_started_at,
                    'completed_at': iter_completed_at,
                    'status': iter_status,
                    'results': iter_results_snapshot,
                })
                # Prevent unbounded growth in memory: keep the latest 25 iterations.
                if len(execution.iterations) > 25:
                    execution.iterations = execution.iterations[-25:]
                execution.last_updated_at = _now_iso()

            _record_event(execution, 'iteration_completed', f"Iteration {iteration} completed", level='info')

            # Decide whether to continue
            if mode == 'once':
                break
            if mode == 'repeat' and planned_iterations is not None and iteration >= planned_iterations:
                break

            # Sleep between iterations
            if mode == 'interval':
                sleep_s = int(interval_minutes * 60)
                _record_event(execution, 'iteration_sleep', f"Sleeping {interval_minutes} minute(s) before next run")
                if not _sleep_cancellable(sleep_s):
                    continue
            elif mode == 'repeat':
                sleep_s = int((repeat_interval_minutes or 0) * 60)
                _record_event(execution, 'iteration_sleep', f"Sleeping {repeat_interval_minutes or 0} minute(s) before next run")
                if not _sleep_cancellable(sleep_s):
                    continue

        # If not cancelled/failed, mark completed.
        with execution_lock:
            if execution.status == 'running':
                execution.status = 'completed'
                _record_event(execution, 'execution_completed', 'Execution completed successfully')
        logger.info(f"\n{'='*60}")
        logger.info(f"WORKFLOW EXECUTION COMPLETED")
        logger.info(f"Status: {execution.status}")
        logger.info(f"Tools executed: {len(execution.results)}")
        logger.info(f"Successful tools: {sum(1 for r in execution.results.values() if r.status == 'completed')}")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        with execution_lock:
            execution.status = 'failed'
        logger.error(f"Workflow execution failed: {e}")
        _record_event(execution, 'execution_failed', f"Workflow execution failed: {e}", level='error')
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        with execution_lock:
            execution.completed_at = datetime.now().isoformat()
        # Ensure no stale active process handle remains.
        with execution_lock:
            active_processes.pop(execution.execution_id, None)
        # Clean up temp directory if enabled
        try:
            if app_settings.get('auto_cleanup', True):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Temp cleanup failed for {temp_dir}: {e}")

# Settings Management Functions
def load_settings():
    """Load application settings from file"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                logger.info(f"Settings loaded from {SETTINGS_FILE}")
                
                # Ensure all default settings are present (backward compatibility)
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                        logger.debug(f"Added missing default setting: {key}")
                
                return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
    
    # Return default settings if file doesn't exist or error
    logger.info("Using default settings")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save application settings to file"""
    try:
        # Only save settings that are in DEFAULT_SETTINGS (filter out any extra)
        valid_settings = {}
        for key in DEFAULT_SETTINGS.keys():
            if key in settings:
                valid_settings[key] = settings[key]
            else:
                valid_settings[key] = DEFAULT_SETTINGS[key]
                logger.warning(f"Missing setting '{key}', using default")
        
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(valid_settings, f, indent=2)
        logger.info(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

# Routes
@app.route('/')
def index():
    """Home page - list workflows"""
    workflows = list(workflows_db.values())
    workflows_dict = [w.to_dict() for w in workflows]
    return render_template('index.html', workflows=workflows_dict)

@app.route('/workflow/create', methods=['GET', 'POST'])
def create_workflow():
    """Create or edit workflow"""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        workflow_id = data.get('id', str(uuid.uuid4()))
        
        # Parse tools from JSON
        tools = []
        for tool_data in data.get('tools', []):
            tools.append(ToolConfig.from_dict(tool_data))
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=data['name'],
            description=data['description'],
            tools=tools,
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=datetime.now().isoformat(),
            run_mode=(data.get('run_mode') or 'once'),
            interval_minutes=_int_or_none(data.get('interval_minutes')),
            repeat_count=_int_or_none(data.get('repeat_count')),
            repeat_interval_minutes=_int_or_none(data.get('repeat_interval_minutes')),
        )
        
        workflows_db[workflow_id] = workflow
        return jsonify({'success': True, 'workflow_id': workflow_id})
    
    workflow_id = request.args.get('id')
    workflow = workflows_db.get(workflow_id)
    workflow_dict = workflow.to_dict() if workflow else None
    app_settings = load_settings()
    return render_template('workflow_editor.html', workflow=workflow_dict,
                           tool_library=app_settings.get('tool_library', []),
                           wordlists=app_settings.get('wordlists', {}))


@app.route('/workflow/delete/<workflow_id>', methods=['POST'])
def delete_workflow(workflow_id):
    """Delete a workflow."""
    if workflow_id in workflows_db:
        del workflows_db[workflow_id]
        if wants_json_response():
            return jsonify({'success': True})
        flash('Workflow deleted.', 'success')
        return redirect(url_for('index'))

    if wants_json_response():
        return jsonify({'success': False, 'error': 'Workflow not found'}), 404
    flash('Workflow not found.', 'error')
    return redirect(url_for('index'))

@app.route('/execute', methods=['POST'])
def execute_workflow():
    """Execute a workflow"""
    data = request.get_json(silent=True) or {}
    workflow_id = data.get('workflow_id')
    domain = data.get('domain')
    
    if not workflow_id:
        return jsonify({'success': False, 'error': 'Workflow ID is required'})
    
    if not domain:
        return jsonify({'success': False, 'error': 'Domain is required'})
    
    workflow = workflows_db.get(workflow_id)
    if not workflow:
        return jsonify({'success': False, 'error': 'Workflow not found'})
    
    # Create execution instance
    execution_id = str(uuid.uuid4())
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        domain=domain,
        status='queued',
        results={},
        run_mode=getattr(workflow, 'run_mode', 'once'),
        interval_minutes=getattr(workflow, 'interval_minutes', None),
        repeat_count=getattr(workflow, 'repeat_count', None),
        repeat_interval_minutes=getattr(workflow, 'repeat_interval_minutes', None),
        current_iteration=0,
        planned_iterations=(getattr(workflow, 'repeat_count', None) if getattr(workflow, 'run_mode', 'once') == 'repeat' else None),
        iterations=[]
    )
    
    execution_history[execution_id] = execution

    # Register cancel event for this execution
    with execution_lock:
        cancel_events[execution_id] = threading.Event()
    
    # Start execution in background thread
    thread = threading.Thread(target=run_workflow_thread, args=(execution,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'execution_id': execution_id,
        'redirect': url_for('execution_status', execution_id=execution_id)
    })

@app.route('/execution/<execution_id>')
def execution_status(execution_id):
    """View execution status and results"""
    execution = execution_history.get(execution_id)
    if not execution:
        return "Execution not found", 404

    workflow = workflows_db.get(execution.workflow_id)

    execution_dict = execution.to_dict(include_results=False)
    execution_dict['workflow_name'] = workflow.name if workflow else "Unknown Workflow"
    execution_dict['workflow_description'] = workflow.description if workflow else ""

    tool_order = []
    if workflow:
        for tool in workflow.tools:
            tool_order.append({
                'id': tool.id,
                'name': tool.name,
                'description': tool.description,
                'color': tool.color,
                'enabled': tool.enabled
            })

    return render_template('execution_detail.html',
                           execution=execution_dict,
                           tool_order=tool_order,
                           settings=load_settings())

@app.route('/execution/<execution_id>/status')
@app.route('/api/execution/<execution_id>/status')
def get_execution_status(execution_id):
    """API endpoint for lightweight real-time status updates (no large outputs)."""
    execution = execution_history.get(execution_id)
    if not execution:
        return jsonify({'error': 'Execution not found'}), 404

    workflow = workflows_db.get(execution.workflow_id)

    # Base metadata without full outputs
    base = execution.to_dict(include_results=False)

    base['workflow_name'] = workflow.name if workflow else "Unknown Workflow"
    base['workflow_description'] = workflow.description if workflow else ""

    # Summarize tool results to keep payload small
    results_summary = {}
    with execution_lock:
        for tool_id, res in (execution.results or {}).items():
            out = res.output or ""
            err = res.error or ""
            results_summary[tool_id] = {
                "tool_id": res.tool_id,
                "tool_name": res.tool_name,
                "status": res.status,
                "start_time": res.start_time,
                "end_time": res.end_time,
                "exit_code": res.exit_code,
                "output_len": len(out),
                "error_len": len(err),
                "output_sha256": _sha256_text(out) if out else "",
                "error_sha256": _sha256_text(err) if err else "",
            }

    base["results_summary"] = results_summary
    base["events_tail"] = (execution.events or [])[-20:]  # last 20 for quick UI

    return jsonify(base)

@app.route('/api/execution/<execution_id>/results/<tool_id>')
def get_execution_tool_results(execution_id, tool_id):
    """Return full stdout/stderr for a single tool (requested on-demand)."""
    execution = execution_history.get(execution_id)
    if not execution:
        return jsonify({'error': 'Execution not found'}), 404

    with execution_lock:
        res = (execution.results or {}).get(tool_id)

    if not res:
        return jsonify({'error': 'Tool result not found'}), 404

    out = res.output or ""
    err = res.error or ""
    return jsonify({
        "execution_id": execution_id,
        "tool_id": tool_id,
        "tool_name": res.tool_name,
        "status": res.status,
        "start_time": res.start_time,
        "end_time": res.end_time,
        "exit_code": res.exit_code,
        "output": out,
        "error": err,
        "output_len": len(out),
        "error_len": len(err),
        "output_sha256": _sha256_text(out) if out else "",
        "error_sha256": _sha256_text(err) if err else "",
    })

@app.route('/api/execution/<execution_id>/events')
def get_execution_events(execution_id):
    """Return the event trail for an execution."""
    execution = execution_history.get(execution_id)
    if not execution:
        return jsonify({'error': 'Execution not found'}), 404
    with execution_lock:
        return jsonify({
            "execution_id": execution_id,
            "version": getattr(execution, "version", 0),
            "events": execution.events or []
        })

@app.route('/img/<path:filename>')
def serve_image(filename):
    """Serve images from the img directory"""
    return send_from_directory('img', filename)

@app.route('/workflow/<workflow_id>/export')
def export_workflow(workflow_id):
    """Export workflow as JSON"""
    workflow = workflows_db.get(workflow_id)
    if not workflow:
        return "Workflow not found", 404
    
    return jsonify(workflow.to_dict())


@app.route('/workflow/import', methods=['POST'])
def import_workflow():
    """Import workflow from JSON."""
    if 'file' not in request.files:
        if wants_json_response():
            return jsonify({'success': False, 'error': 'No file uploaded'})
        flash('No file uploaded.', 'error')
        return redirect(url_for('create_workflow'))

    file = request.files['file']
    if file.filename == '':
        if wants_json_response():
            return jsonify({'success': False, 'error': 'No file selected'})
        flash('No file selected.', 'error')
        return redirect(url_for('create_workflow'))

    try:
        data = json.load(file)
        tools = []
        for tool_data in data.get('tools', []):
            tools.append(ToolConfig.from_dict(tool_data))

        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=data.get('name', 'Imported Workflow'),
            description=data.get('description', ''),
            tools=tools,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            run_mode=(data.get('run_mode') or 'once'),
            interval_minutes=_int_or_none(data.get('interval_minutes')),
            repeat_count=_int_or_none(data.get('repeat_count')),
            repeat_interval_minutes=_int_or_none(data.get('repeat_interval_minutes')),
        )
        workflows_db[workflow.id] = workflow

        if wants_json_response():
            return jsonify({'success': True, 'workflow': workflow.to_dict()})

        flash('Workflow imported successfully.', 'success')
        return redirect(url_for('create_workflow', id=workflow.id))

    except Exception as e:
        if wants_json_response():
            return jsonify({'success': False, 'error': str(e)})
        flash(f'Failed to import workflow: {e}', 'error')
        return redirect(url_for('create_workflow'))

@app.route('/executions')
def executions_list():
    """List all executions"""
    executions = list(execution_history.values())
    executions_dict = []
    
    for execution in executions:
        exec_dict = execution.to_dict()
        workflow = workflows_db.get(execution.workflow_id)
        exec_dict['workflow_name'] = workflow.name if workflow else "Unknown Workflow"
        exec_dict['workflow_description'] = workflow.description if workflow else ""

        # Compute duration for completed executions (avoid doing this in templates)
        def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                # Best-effort: strip timezone/Z if present
                try:
                    return datetime.fromisoformat(ts.replace('Z', '').split('+')[0])
                except Exception:
                    return None

        def _human_duration(seconds: Optional[int]) -> str:
            if seconds is None:
                return "--"
            if seconds < 0:
                seconds = 0
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            if h > 0:
                return f"{h}h {m}m {s}s"
            if m > 0:
                return f"{m}m {s}s"
            return f"{s}s"

        start_dt = _parse_iso(exec_dict.get('started_at'))
        end_dt = _parse_iso(exec_dict.get('completed_at'))
        if start_dt and end_dt:
            duration_seconds = int((end_dt - start_dt).total_seconds())
        else:
            duration_seconds = None
        exec_dict['duration_seconds'] = duration_seconds
        exec_dict['duration_human'] = _human_duration(duration_seconds)
        
        # Calculate summary statistics
        total_tools = len(workflow.tools) if workflow else 0
        completed_tools = sum(1 for r in execution.results.values() if r.status == 'completed')
        exec_dict['summary'] = {
            'total_tools': total_tools,
            'completed_tools': completed_tools,
            'success_rate': (completed_tools / total_tools * 100) if total_tools > 0 else 0
        }
        
        executions_dict.append(exec_dict)
    
    executions_dict.sort(key=lambda x: x['created_at'], reverse=True)
    
    return render_template('executions.html', executions=executions_dict)

@app.route('/execution/delete/<execution_id>', methods=['POST'])
def delete_execution(execution_id):
    """Delete an execution"""
    if execution_id in execution_history:
        # Best-effort: cancel and terminate any running subprocess first.
        with execution_lock:
            evt = cancel_events.get(execution_id)
            if evt:
                evt.set()
            proc = active_processes.get(execution_id)
        if proc is not None:
            _terminate_process_tree(proc)

        with execution_lock:
            execution_history.pop(execution_id, None)
            cancel_events.pop(execution_id, None)
            active_processes.pop(execution_id, None)

        if wants_json_response():
            return jsonify({'success': True})
        flash('Execution deleted.', 'success')
        return redirect(url_for('executions_list'))

    if wants_json_response():
        return jsonify({'success': False, 'error': 'Execution not found'}), 404
    flash('Execution not found.', 'error')
    return redirect(url_for('executions_list'))


@app.route('/execution/cancel/<execution_id>', methods=['POST'])
def cancel_execution(execution_id):
    """Cancel a running execution (best-effort)."""
    execution = execution_history.get(execution_id)
    if not execution:
        if wants_json_response():
            return jsonify({'success': False, 'error': 'Execution not found'}), 404
        flash('Execution not found.', 'error')
        return redirect(url_for('executions_list'))

    with execution_lock:
        evt = cancel_events.get(execution_id)
        if evt is None:
            evt = threading.Event()
            cancel_events[execution_id] = evt
        evt.set()
        execution.cancel_requested = True
        if execution.status in ('queued', 'running'):
            execution.status = 'cancelling'
        execution.last_updated_at = _now_iso()
        proc = active_processes.get(execution_id)

    if proc is not None:
        _terminate_process_tree(proc)

    _record_event(execution, 'cancel_requested', 'Cancel requested by user')

    if wants_json_response():
        return jsonify({'success': True, 'status': execution.status})
    flash('Cancel requested.', 'info')
    return redirect(url_for('execution_status', execution_id=execution_id))

@app.route('/execution/export/<execution_id>')
def export_execution(execution_id):
    """Export execution results as JSON"""
    execution = execution_history.get(execution_id)
    if not execution:
        return "Execution not found", 404
    
    export_data = execution.to_dict()
    
    workflow = workflows_db.get(execution.workflow_id)
    if workflow:
        export_data['workflow_info'] = {
            'name': workflow.name,
            'description': workflow.description
        }
    
    filename = f"execution_{execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('exports', filename)
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/execution/clear_all', methods=['POST'])
def clear_all_executions():
    """Clear all execution history"""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        confirmation = data.get('confirm', False)
        if confirmation:
            # Best-effort: stop any running executions first.
            with execution_lock:
                for exec_id, proc in list(active_processes.items()):
                    try:
                        evt = cancel_events.get(exec_id)
                        if evt:
                            evt.set()
                    except Exception:
                        pass
                for proc in list(active_processes.values()):
                    _terminate_process_tree(proc)
                active_processes.clear()
                cancel_events.clear()
                execution_history.clear()
            return jsonify({'success': True, 'message': 'All executions cleared'})
        return jsonify({'success': False, 'error': 'Confirmation required'})
    return jsonify({'success': False, 'error': 'Invalid request'})

# Settings Routes
@app.route('/settings')
def settings():
    """Settings page"""
    app_settings = load_settings()

    return render_template("settings.html", settings=app_settings)

@app.route('/settings/update', methods=['POST'])
def update_settings():
    """Update application settings"""
    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) or {}
            # Backward/forward compatibility: accept 'notifications' as alias for 'enable_notifications'
            if isinstance(data, dict) and 'notifications' in data and 'enable_notifications' not in data:
                data['enable_notifications'] = data.pop('notifications')

            logger.info(f"Received settings update request")
            
            # Load current settings
            current_settings = load_settings()
            
            # Update with new values
            updated_count = 0
            for key, value in data.items():
                if key in DEFAULT_SETTINGS:
                    # Validate and convert types if needed
                    if key in ['max_execution_time', 'max_output_size', 'concurrent_executions', 'log_retention_days']:
                        try:
                            value = int(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid value for {key}: {value}, using default")
                            value = DEFAULT_SETTINGS[key]
                    
                    if key in ['auto_cleanup', 'enable_logging', 'enable_notifications', 'enable_auto_updates', 'enable_debug_mode']:
                        if isinstance(value, str):
                            value = value.lower() in ['true', 'yes', '1', 'on']
                    

                    if key == 'tool_library':
                        # Accept list of tool definitions; if provided as JSON string, parse it
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except Exception:
                                logger.warning("Invalid JSON for tool_library; using existing value")
                                value = current_settings.get('tool_library', DEFAULT_SETTINGS.get('tool_library', []))
                        if not isinstance(value, list):
                            logger.warning("tool_library must be a list; using existing value")
                            value = current_settings.get('tool_library', DEFAULT_SETTINGS.get('tool_library', []))

                    if key == 'wordlists':
                        # Accept dict of {name: path}; if provided as JSON string, parse it
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except Exception:
                                logger.warning("Invalid JSON for wordlists; using existing value")
                                value = current_settings.get('wordlists', DEFAULT_SETTINGS.get('wordlists', {}))
                        if not isinstance(value, dict):
                            logger.warning("wordlists must be a dict; using existing value")
                            value = current_settings.get('wordlists', DEFAULT_SETTINGS.get('wordlists', {}))

                    current_settings[key] = value
                    updated_count += 1
                    logger.debug(f"Updated setting '{key}' to '{value}'")
                else:
                    logger.warning(f"Ignoring unknown setting key: '{key}'")
            
            # Save updated settings
            if save_settings(current_settings):
                return jsonify({
                    'success': True,
                    'message': f'{updated_count} settings updated successfully',
                    'settings': current_settings
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to save settings'})
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid request'})


@app.route('/settings/backup', methods=['GET', 'POST'])
def backup_settings():
    """Create a backup (POST) or download a backup file (GET)."""
    os.makedirs('backups', exist_ok=True)

    # Download existing backup
    download_name = request.args.get('download')
    if request.method == 'GET':
        if not download_name:
            # The UI should trigger POST to create; GET without download is not supported.
            return jsonify({'success': False, 'error': 'Missing download parameter'}), 400

        safe_name = os.path.basename(download_name)
        filepath = os.path.join('backups', safe_name)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Backup file not found'}), 404

        return send_file(filepath, as_attachment=True, download_name=safe_name)

    # Create new backup
    backup_data = {
        'workflows': [w.to_dict() for w in workflows_db.values()],
        'executions': [e.to_dict() for e in execution_history.values()],
        'settings': load_settings(),
        'backup_date': datetime.now().isoformat(),
        'version': '1.0.0'
    }

    filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('backups', filename)

    with open(filepath, 'w') as f:
        json.dump(backup_data, f, indent=2)

    # If called from a browser form, redirect to download
    if not wants_json_response():
        flash('Backup created successfully.', 'success')
        return redirect(url_for('backup_settings') + f"?download={filename}")

    return jsonify({'success': True, 'filename': filename})


@app.route('/settings/restore', methods=['POST'])
def restore_settings():
    """Restore from backup.

    For browser form submissions, redirect back to Settings with a flash message.
    For AJAX/API calls, return JSON.
    """
    if 'file' not in request.files:
        if wants_json_response():
            return jsonify({'success': False, 'error': 'No file uploaded'})
        flash('No file uploaded.', 'error')
        return redirect(url_for('settings'))

    file = request.files['file']
    if file.filename == '':
        if wants_json_response():
            return jsonify({'success': False, 'error': 'No file selected'})
        flash('No file selected.', 'error')
        return redirect(url_for('settings'))

    try:
        data = json.load(file)

        # Restore settings (Tool Library / Wordlists, etc.)
        # Backup schema stores settings under top-level key 'settings'.
        backup_settings = data.get('settings', {})
        if isinstance(backup_settings, dict) and backup_settings:
            current_settings = load_settings()
            # Merge: prefer backup values when present.
            for k in DEFAULT_SETTINGS.keys():
                if k in backup_settings:
                    current_settings[k] = backup_settings[k]
            save_settings(current_settings)

        workflows_db.clear()
        execution_history.clear()

        # Restore workflows
        for workflow_data in data.get('workflows', []):
            tools = []
            for tool_data in workflow_data.get('tools', []):
                tools.append(ToolConfig.from_dict(tool_data))

            workflow = Workflow(
                workflow_id=workflow_data.get('id', str(uuid.uuid4())),
                name=workflow_data['name'],
                description=workflow_data.get('description', ''),
                tools=tools,
                created_at=workflow_data.get('created_at', datetime.now().isoformat()),
                updated_at=datetime.now().isoformat(),
                run_mode=(workflow_data.get('run_mode') or 'once'),
                interval_minutes=_int_or_none(workflow_data.get('interval_minutes')),
                repeat_count=_int_or_none(workflow_data.get('repeat_count')),
                repeat_interval_minutes=_int_or_none(workflow_data.get('repeat_interval_minutes'))
            )
            workflows_db[workflow.id] = workflow

        # Restore executions if requested
        # IMPORTANT: this must be a data import only. Never restart/resume executions.
        restore_executions = request.form.get('restore_executions', 'false').lower() == 'true'
        if restore_executions:
            for exec_data in data.get('executions', []):
                restored_status = exec_data.get('status', 'completed')
                # If the backup contains in-flight states, mark as restored to avoid any future "resume" logic.
                if restored_status in ('queued', 'running'):
                    restored_status = 'restored'

                results = {}
                for k, v in (exec_data.get('results') or {}).items():
                    try:
                        results[k] = ExecutionResult(
                            tool_id=v.get('tool_id', k),
                            tool_name=v.get('tool_name', ''),
                            status=v.get('status', 'completed'),
                            start_time=v.get('start_time'),
                            end_time=v.get('end_time'),
                            output=v.get('output'),
                            error=v.get('error'),
                            exit_code=v.get('exit_code')
                        )
                    except Exception:
                        continue

                execution = WorkflowExecution(
                    execution_id=exec_data.get('execution_id', str(uuid.uuid4())),
                    workflow_id=exec_data.get('workflow_id'),
                    domain=exec_data.get('domain', ''),
                    status=restored_status,
                    results=results,
                    created_at=exec_data.get('created_at'),
                    started_at=exec_data.get('started_at'),
                    completed_at=exec_data.get('completed_at'),
                    version=exec_data.get('version', 0),
                    last_updated_at=exec_data.get('last_updated_at'),
                    events=exec_data.get('events', []),
                    run_mode=(exec_data.get('run_mode') or 'once'),
                    interval_minutes=_int_or_none(exec_data.get('interval_minutes')),
                    repeat_count=_int_or_none(exec_data.get('repeat_count')),
                    repeat_interval_minutes=_int_or_none(exec_data.get('repeat_interval_minutes')),
                    current_iteration=_int_or_none(exec_data.get('current_iteration')) or 0,
                    planned_iterations=_int_or_none(exec_data.get('planned_iterations')),
                    iterations=exec_data.get('iterations', [])
                )
                execution_history[execution.execution_id] = execution

        if not wants_json_response():
            flash('Backup restored successfully.', 'success')
            return redirect(url_for('index'))

        return jsonify({'success': True, 'message': 'Backup restored successfully'})

    except Exception as e:
        if wants_json_response():
            return jsonify({'success': False, 'error': str(e)})
        flash(f'Failed to restore backup: {e}', 'error')
        return redirect(url_for('settings'))

@app.route('/tools/update')
def tools_update():
    """Tools update management page"""
    settings = load_settings()
    all_tools = []

    # 1) Tool Library tools (global)
    for entry in (settings.get('tool_library', []) or []):
        all_tools.append({
            'id': entry.get('id'),
            'name': entry.get('name', 'Unnamed'),
            # Keep legacy key names expected by the template
            'command': entry.get('path', entry.get('command', '')),
            'args_template': entry.get('default_command', entry.get('args_template', '')),
            'update_command': entry.get('update_command', ''),
            'last_updated': entry.get('last_updated'),
            'description': entry.get('description', ''),
            'color': entry.get('color', '#4a86e8'),
            'workflow_id': '__library__',
            'workflow_name': 'Tool Library'
        })

    # 2) Workflow-defined tools (legacy)
    for workflow_id, workflow in workflows_db.items():
        for tool in workflow.tools:
            tool_dict = tool.to_dict()
            tool_dict['workflow_id'] = workflow_id
            tool_dict['workflow_name'] = workflow.name
            all_tools.append(tool_dict)
    
    recent_updates = list(update_history.values())[-10:]
    recent_updates_dict = [u.to_dict() for u in recent_updates]
    
    return render_template('tools_update.html', 
                         tools=all_tools, 
                         recent_updates=recent_updates_dict)

@app.route('/api/tool/update', methods=['POST'])
def update_tool():
    """Update a specific tool"""
    data = request.get_json(silent=True) or {}
    tool_id = data.get('tool_id')
    workflow_id = data.get('workflow_id')

    if not tool_id or not workflow_id:
        return jsonify({'success': False, 'error': 'Missing tool_id or workflow_id'})

    # Tool Library update
    if workflow_id == '__library__':
        settings = load_settings()
        entry = None
        for t in (settings.get('tool_library', []) or []):
            if t.get('id') == tool_id:
                entry = t
                break
        if not entry:
            return jsonify({'success': False, 'error': 'Tool not found in Tool Library'})

        target_tool = ToolConfig(
            tool_id=entry.get('id', str(uuid.uuid4())),
            name=entry.get('name', 'Unnamed'),
            command=entry.get('path', entry.get('command', '')),
            args_template=entry.get('default_command', entry.get('args_template', '')),
            update_command=entry.get('update_command', ''),
            description=entry.get('description', ''),
        )
    else:
        workflow = workflows_db.get(workflow_id)
        if not workflow:
            return jsonify({'success': False, 'error': 'Workflow not found'})

        target_tool = None
        for tool in workflow.tools:
            if tool.id == tool_id:
                target_tool = tool
                break
        if not target_tool:
            return jsonify({'success': False, 'error': 'Tool not found in workflow'})
    
    if not target_tool.update_command:
        return jsonify({'success': False, 'error': 'No update command configured for this tool'})
    
    thread = threading.Thread(target=execute_update_thread, args=(target_tool, workflow_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Update started for {target_tool.name}',
        'tool_name': target_tool.name
    })

@app.route('/api/tools/update-all', methods=['POST'])
def update_all_tools():
    """Update all tools with update commands"""
    data = request.get_json(silent=True) or {}
    workflow_id = data.get('workflow_id')
    
    tools_to_update = []
    
    settings = load_settings()

    if workflow_id == '__library__':
        # Only library tools
        for entry in (settings.get('tool_library', []) or []):
            if entry.get('update_command'):
                tools_to_update.append((ToolConfig(
                    tool_id=entry.get('id', str(uuid.uuid4())),
                    name=entry.get('name', 'Unnamed'),
                    command=entry.get('path', entry.get('command', '')),
                    args_template=entry.get('default_command', entry.get('args_template', '')),
                    update_command=entry.get('update_command', ''),
                    description=entry.get('description', ''),
                ), '__library__'))
    elif workflow_id:
        # Only a specific workflow
        workflow = workflows_db.get(workflow_id)
        if workflow:
            for tool in workflow.tools:
                if tool.update_command:
                    tools_to_update.append((tool, workflow_id))
    else:
        # All tools: library + all workflows
        for entry in (settings.get('tool_library', []) or []):
            if entry.get('update_command'):
                tools_to_update.append((ToolConfig(
                    tool_id=entry.get('id', str(uuid.uuid4())),
                    name=entry.get('name', 'Unnamed'),
                    command=entry.get('path', entry.get('command', '')),
                    args_template=entry.get('default_command', entry.get('args_template', '')),
                    update_command=entry.get('update_command', ''),
                    description=entry.get('description', ''),
                ), '__library__'))

        for wf_id, workflow in workflows_db.items():
            for tool in workflow.tools:
                if tool.update_command:
                    tools_to_update.append((tool, wf_id))
    
    if not tools_to_update:
        return jsonify({'success': False, 'error': 'No tools with update commands found'})
    
    update_ids = []
    for tool, wf_id in tools_to_update:
        update_id = str(uuid.uuid4())
        result = UpdateResult(
            update_id=update_id,
            tool_id=tool.id,
            tool_name=tool.name,
            status='queued',
            start_time=datetime.now().isoformat()
        )
        update_history[update_id] = result
        
        thread = threading.Thread(target=execute_update_thread, args=(tool, wf_id))
        thread.daemon = True
        thread.start()
        
        update_ids.append(update_id)
    
    return jsonify({
        'success': True,
        'message': f'Started updates for {len(tools_to_update)} tools',
        'update_count': len(tools_to_update),
        'update_ids': update_ids
    })

@app.route('/api/update/status/<update_id>')
def get_update_status(update_id):
    """Get status of an update"""
    update = update_history.get(update_id)
    if not update:
        return jsonify({'error': 'Update not found'}), 404
    
    return jsonify(update.to_dict())

@app.route('/api/updates/recent')
def get_recent_updates():
    """Get recent update history"""
    updates = list(update_history.values())[-20:]
    updates_dict = [u.to_dict() for u in updates]
    return jsonify(updates_dict)

# Load exported workflow function
def load_exported_workflow():
    """Load an exported workflow from JSON file"""
    try:
        with open('export.json', 'r') as f:
            data = json.load(f)
        
        tools = []
        for tool_data in data.get('tools', []):
            tools.append(ToolConfig.from_dict(tool_data))
        
        workflow = Workflow(
            workflow_id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            description=data['description'],
            tools=tools,
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=datetime.now().isoformat()
        )
        
        workflows_db[workflow.id] = workflow
        print(f"✓ Loaded exported workflow: {workflow.name}")
        return True
    except Exception as e:
        print(f"✗ Error loading exported workflow: {e}")
        return False

# Initialize with sample workflow
def create_sample_workflow():
    """Create a sample workflow for demonstration"""
    sample_workflow = Workflow(
        workflow_id="sample-workflow",
        name="Example Recon Workflow",
        description="Example workflow with proper tool chaining",
        tools=[
            ToolConfig(
                tool_id="tool_assetfinder",
                name="Assetfinder",
                command="/tools/assetfinder",
                args_template="{domain}",
                input_source="initial",
                input_method="argument",
                output_handling="stdout",
                provides_output=True,
                output_format="list",
                placeholder_name="subdomains",
                description="Find subdomains",
                color="#4a86e8",
                update_command="--update"
            ),
            ToolConfig(
                tool_id="tool_nuclei",
                name="Nuclei Scanner",
                command="/tools/nuclei",
                args_template="-l {subdomains}",
                input_source="previous",
                input_method="file",
                output_handling="stdout",
                provides_output=True,
                output_format="text",
                placeholder_name="scan_results",
                description="Scan discovered assets",
                color="#5cb85c",
                update_command="-update"
            ),
            ToolConfig(
                tool_id="tool_notify",
                name="Notification",
                command="echo",
                args_template="Scan completed for {domain}",
                input_source="none",
                input_method="none",
                output_handling="stdout",
                provides_output=False,
                description="Final notification",
                color="#f0ad4e"
            )
        ],
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    workflows_db[sample_workflow.id] = sample_workflow
    print("✓ Created enhanced sample workflow")

if __name__ == '__main__':
    create_sample_workflow()
    
    # Try to load exported workflow if it exists
    if os.path.exists('export.json'):
        load_exported_workflow()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
