"""Communication layer between simulator and controller."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MessageType(Enum):
    """Types of messages in the communication protocol."""

    # Simulator -> Controller
    OBSERVATION = "observation"
    STATE_REQUEST = "state_request"
    RESET = "reset"
    ERROR = "error"

    # Controller -> Simulator
    CONTROL_COMMAND = "control_command"
    STATE_REPORT = "state_report"


@dataclass
class Message:
    """Message passed between simulator and controller."""

    message_type: MessageType
    timestamp: float
    sender: str
    receiver: str
    payload: dict[str, Any] = field(default_factory=dict)
    sequence_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self._serialize_payload(self.payload),
            "sequence_id": self.sequence_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                result[key] = float(value)
            else:
                result[key] = value
        return result


class CommunicationLayer:
    """Communication layer using queue.Queue for thread-safe message passing.

    Two queues:
        sim_to_ctrl  — simulator sends observations/resets/errors -> controller reads
        ctrl_to_sim  — controller sends commands/reports -> simulator reads
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._sim_to_ctrl: queue.Queue[Message] = queue.Queue(maxsize=maxsize)
        self._ctrl_to_sim: queue.Queue[Message] = queue.Queue(maxsize=maxsize)
        self._message_counter = 0
        self._message_log: list[Message] = []
        self._max_log_size = 1000

    # ------------------------------------------------------------------
    # Simulator -> Controller
    # ------------------------------------------------------------------

    def send_observation(self, timestamp: float, observation: dict[str, Any]) -> None:
        """Simulator sends current observation to controller."""
        self._put(
            self._sim_to_ctrl,
            Message(
                message_type=MessageType.OBSERVATION,
                timestamp=timestamp,
                sender="simulator",
                receiver="controller",
                payload={"observation": observation},
            ),
        )

    def receive_observation(self) -> dict[str, Any] | None:
        """Controller reads latest observation (non-blocking)."""
        msg = self._get(self._sim_to_ctrl, MessageType.OBSERVATION)
        return msg.payload.get("observation") if msg else None

    def send_reset(self, timestamp: float) -> None:
        """Simulator signals controller to reset."""
        self._put(
            self._sim_to_ctrl,
            Message(
                message_type=MessageType.RESET,
                timestamp=timestamp,
                sender="simulator",
                receiver="controller",
                payload={},
            ),
        )

    def receive_reset(self) -> bool:
        """Controller checks for reset signal."""
        return self._get(self._sim_to_ctrl, MessageType.RESET) is not None

    def send_error(self, timestamp: float, error_message: str) -> None:
        """Send error notification."""
        self._put(
            self._sim_to_ctrl,
            Message(
                message_type=MessageType.ERROR,
                timestamp=timestamp,
                sender="system",
                receiver="all",
                payload={"error_message": error_message},
            ),
        )

    def receive_error(self) -> str | None:
        """Read error message if any."""
        msg = self._get(self._sim_to_ctrl, MessageType.ERROR)
        return msg.payload.get("error_message") if msg else None

    # ------------------------------------------------------------------
    # Controller -> Simulator
    # ------------------------------------------------------------------

    def send_control_command(
        self,
        timestamp: float,
        control_output: np.ndarray,
        controller_state: str,
    ) -> None:
        """Controller sends motor commands to simulator."""
        self._put(
            self._ctrl_to_sim,
            Message(
                message_type=MessageType.CONTROL_COMMAND,
                timestamp=timestamp,
                sender="controller",
                receiver="simulator",
                payload={
                    "control_output": control_output,
                    "controller_state": controller_state,
                },
            ),
        )

    def receive_control_command(self) -> tuple[np.ndarray, str] | None:
        """Simulator reads latest control command."""
        msg = self._get(self._ctrl_to_sim, MessageType.CONTROL_COMMAND)
        if msg:
            return msg.payload.get("control_output"), msg.payload.get("controller_state")
        return None

    def send_state_report(self, timestamp: float, controller_state: dict[str, Any]) -> None:
        """Controller sends its internal state for logging/debugging."""
        self._put(
            self._ctrl_to_sim,
            Message(
                message_type=MessageType.STATE_REPORT,
                timestamp=timestamp,
                sender="controller",
                receiver="simulator",
                payload={"state_info": controller_state},
            ),
        )

    def receive_state_report(self) -> dict[str, Any] | None:
        """Read state report if available."""
        msg = self._get(self._ctrl_to_sim, MessageType.STATE_REPORT)
        return msg.payload.get("state_info") if msg else None

    # ------------------------------------------------------------------
    # Stats & log
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "sim_to_ctrl_size": self._sim_to_ctrl.qsize(),
            "ctrl_to_sim_size": self._ctrl_to_sim.qsize(),
            "total_messages": self._message_counter,
            "log_size": len(self._message_log),
        }

    def get_message_log(self, last_n: int = 50) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._message_log[-last_n:]]

    def clear_log(self) -> None:
        self._message_log.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _put(self, q: queue.Queue[Message], message: Message) -> None:
        """Assign sequence id, log, and enqueue (drop oldest if full)."""
        message.sequence_id = self._message_counter
        self._message_counter += 1
        self._log(message)
        try:
            q.put_nowait(message)
        except queue.Full:
            q.get_nowait()  # drop oldest to make room
            q.put_nowait(message)

    def _get(self, q: queue.Queue[Message], message_type: MessageType) -> Message | None:
        """Drain the queue looking for the first message of the given type.

        Messages of other types are put back (in order) so they are not lost.
        Non-blocking: returns None immediately if nothing matches.
        """
        found: Message | None = None
        skipped: list[Message] = []

        while not q.empty():
            try:
                msg = q.get_nowait()
            except queue.Empty:
                break
            if msg.message_type == message_type and found is None:
                found = msg
            else:
                skipped.append(msg)

        for msg in skipped:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # drop if buffer overflowed

        return found

    def _log(self, message: Message) -> None:
        self._message_log.append(message)
        if len(self._message_log) > self._max_log_size:
            self._message_log.pop(0)
