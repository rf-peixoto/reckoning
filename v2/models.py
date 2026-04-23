"""
models.py – Plain Python data classes.
No database logic here; see database.py.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


class ToolConfig:
    """Configuration for a single tool/step inside a workflow."""

    def __init__(
        self,
        tool_id: Optional[str] = None,
        name: str = "",
        command: str = "",
        args_template: str = "",
        input_source: str = "specific",
        library_tool_id: Optional[str] = None,
        command_override: Optional[str] = None,
        args_template_override: Optional[str] = None,
        update_command_override: Optional[str] = None,
        description_override: Optional[str] = None,
        specific_step: Optional[int] = None,
        description: str = "",
        color: str = "#00ff41",
        enabled: bool = True,
        update_command: str = "",
        last_updated: Optional[str] = None,
        input_method: str = "argument",
        output_handling: str = "stdout",
        provides_output: bool = True,
        output_format: str = "text",
        output_file_path: str = "",
        placeholder_name: str = "input",
        timeout_override: Optional[int] = None,  # per-step timeout (seconds)
    ):
        self.id = tool_id or str(uuid.uuid4())
        self.name = name
        self.library_tool_id = library_tool_id
        self.command_override = command_override
        self.args_template_override = args_template_override
        self.update_command_override = update_command_override
        self.description_override = description_override
        self.command = command
        self.args_template = args_template
        self.input_source = input_source
        self.specific_step = specific_step
        self.description = description
        self.color = color
        self.enabled = enabled
        self.update_command = update_command or ""
        self.last_updated = last_updated
        self.input_method = input_method
        self.output_handling = output_handling
        self.provides_output = provides_output
        self.output_format = output_format
        self.output_file_path = output_file_path
        self.placeholder_name = placeholder_name
        self.timeout_override = timeout_override

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "command": self.command,
            "args_template": self.args_template,
            "library_tool_id": self.library_tool_id,
            "command_override": self.command_override,
            "args_template_override": self.args_template_override,
            "update_command_override": self.update_command_override,
            "description_override": self.description_override,
            "input_source": self.input_source,
            "specific_step": self.specific_step,
            "description": self.description,
            "color": self.color,
            "enabled": self.enabled,
            "update_command": self.update_command,
            "last_updated": self.last_updated,
            "input_method": self.input_method,
            "output_handling": self.output_handling,
            "provides_output": self.provides_output,
            "output_format": self.output_format,
            "output_file_path": self.output_file_path,
            "placeholder_name": self.placeholder_name,
            "timeout_override": self.timeout_override,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolConfig":
        return cls(
            tool_id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unnamed"),
            command=data.get("command", ""),
            args_template=data.get("args_template", ""),
            input_source=data.get("input_source", "specific"),
            library_tool_id=data.get("library_tool_id"),
            command_override=data.get("command_override"),
            args_template_override=data.get("args_template_override"),
            update_command_override=data.get("update_command_override"),
            description_override=data.get("description_override"),
            specific_step=data.get("specific_step"),
            description=data.get("description", ""),
            color=data.get("color", "#00ff41"),
            enabled=data.get("enabled", True),
            update_command=data.get("update_command", ""),
            last_updated=data.get("last_updated"),
            input_method=data.get("input_method", "argument"),
            output_handling=data.get("output_handling", "stdout"),
            provides_output=data.get("provides_output", True),
            output_format=data.get("output_format", "text"),
            output_file_path=data.get("output_file_path", ""),
            placeholder_name=data.get("placeholder_name", "input"),
            timeout_override=data.get("timeout_override"),
        )


class Workflow:
    """Complete workflow definition."""

    VALID_RUN_MODES = {"once", "interval", "repeat", "scheduled"}

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        name: str = "",
        description: str = "",
        tools: Optional[List[ToolConfig]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        author: str = "anonymous",
        run_mode: str = "once",
        interval_minutes: Optional[int] = None,
        repeat_count: Optional[int] = None,
        repeat_interval_minutes: Optional[int] = None,
        scheduled_at: Optional[str] = None,
    ):
        self.id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tools = tools or []
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.author = author
        raw_mode = (run_mode or "once").strip().lower()
        self.run_mode = raw_mode if raw_mode in self.VALID_RUN_MODES else "once"
        self.interval_minutes = interval_minutes
        self.repeat_count = repeat_count
        self.repeat_interval_minutes = repeat_interval_minutes
        self.scheduled_at = scheduled_at  # ISO datetime string for 'scheduled' mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tools": [t.to_dict() for t in self.tools],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "run_mode": self.run_mode,
            "interval_minutes": self.interval_minutes,
            "repeat_count": self.repeat_count,
            "repeat_interval_minutes": self.repeat_interval_minutes,
            "scheduled_at": self.scheduled_at,
        }


class ExecutionResult:
    """Result snapshot for a single tool step."""

    def __init__(
        self,
        tool_id: str,
        tool_name: str,
        status: str = "pending",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        output: str = "",
        error: str = "",
        exit_code: Optional[int] = None,
    ):
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.output = output
        self.error = error
        self.exit_code = exit_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            tool_id=data.get("tool_id", ""),
            tool_name=data.get("tool_name", ""),
            status=data.get("status", "pending"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            output=data.get("output", ""),
            error=data.get("error", ""),
            exit_code=data.get("exit_code"),
        )


class WorkflowExecution:
    """A single execution instance of a workflow."""

    def __init__(
        self,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        domain: str = "",
        notes: str = "",
        status: str = "queued",
        results: Optional[Dict[str, ExecutionResult]] = None,
        created_at: Optional[str] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        version: int = 0,
        last_updated_at: Optional[str] = None,
        events: Optional[List[Dict]] = None,
        cancel_requested: bool = False,
        cancelled_at: Optional[str] = None,
        run_mode: str = "once",
        interval_minutes: Optional[int] = None,
        repeat_count: Optional[int] = None,
        repeat_interval_minutes: Optional[int] = None,
        scheduled_at: Optional[str] = None,
        current_iteration: int = 0,
        planned_iterations: Optional[int] = None,
        iterations: Optional[List[Dict]] = None,
    ):
        self.execution_id = execution_id or str(uuid.uuid4())
        self.workflow_id = workflow_id
        self.domain = domain
        self.notes = notes or ""
        self.status = status
        self.results: Dict[str, ExecutionResult] = results or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.started_at = started_at
        self.completed_at = completed_at
        self.version = int(version or 0)
        self.last_updated_at = last_updated_at or self.created_at
        self.events: List[Dict] = events or []
        self.cancel_requested = bool(cancel_requested)
        self.cancelled_at = cancelled_at
        self.run_mode = run_mode or "once"
        self.interval_minutes = interval_minutes
        self.repeat_count = repeat_count
        self.repeat_interval_minutes = repeat_interval_minutes
        self.scheduled_at = scheduled_at
        self.current_iteration = int(current_iteration or 0)
        self.planned_iterations = planned_iterations
        self.iterations: List[Dict] = iterations or []

    def to_dict(self, include_results: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "domain": self.domain,
            "notes": self.notes,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "version": self.version,
            "last_updated_at": self.last_updated_at,
            "events": self.events,
            "cancel_requested": self.cancel_requested,
            "cancelled_at": self.cancelled_at,
            "run_mode": self.run_mode,
            "interval_minutes": self.interval_minutes,
            "repeat_count": self.repeat_count,
            "repeat_interval_minutes": self.repeat_interval_minutes,
            "scheduled_at": self.scheduled_at,
            "current_iteration": self.current_iteration,
            "planned_iterations": self.planned_iterations,
            "iterations": self.iterations,
        }
        if include_results:
            data["results"] = {k: v.to_dict() for k, v in (self.results or {}).items()}
        return data


class UpdateResult:
    """Result of a tool update operation."""

    def __init__(
        self,
        update_id: Optional[str] = None,
        tool_id: str = "",
        tool_name: str = "",
        status: str = "pending",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        output: str = "",
        error: str = "",
    ):
        self.update_id = update_id or str(uuid.uuid4())
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.output = output
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "tool_id": self.tool_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output": self.output,
            "error": self.error,
        }
