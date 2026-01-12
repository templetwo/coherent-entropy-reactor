import asyncio
import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import yaml

from src.core.reactor import CoherentEntropyReactor

app = FastAPI()

# Load configuration
CONFIG_PATH = Path("config/reactor_config.yaml")

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return None

config = load_config()

# Initialize Reactor with config or defaults
reactor_config = config['reactor'] if config else {}
reactor = CoherentEntropyReactor(
    input_dim=reactor_config.get('input_dim', 128),
    hidden_dim=reactor_config.get('hidden_dim', 256),
    output_dim=reactor_config.get('output_dim', 128),
    num_layers=reactor_config.get('num_layers', 2),
    kuramoto_k=reactor_config.get('kuramoto', {}).get('coupling_strength', 2.0),
    target_entropy=reactor_config.get('target_entropy', 3.0)
)

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

class SessionRecorder:
    def __init__(self):
        self.is_recording = False
        self.session_name = ""
        self.data = []
        self.annotations = []
        self.start_time = 0

    def start(self, name: str):
        self.is_recording = True
        self.session_name = name
        self.data = []
        self.annotations = []
        self.start_time = asyncio.get_event_loop().time()

    def record_step(self, metrics: dict):
        if self.is_recording:
            m_copy = metrics.copy()
            m_copy["timestamp"] = asyncio.get_event_loop().time() - self.start_time
            self.data.append(m_copy)

    def annotate(self, note: str, marker_type: str = "manual"):
        if self.is_recording:
            self.annotations.append({
                "timestamp": asyncio.get_event_loop().time() - self.start_time,
                "note": note,
                "marker": marker_type
            })

    def stop(self) -> str:
        if not self.is_recording:
            return ""
        self.is_recording = False
        filename = f"session_{self.session_name}_{int(self.start_time)}.json"
        filepath = Path("experiments/sessions") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({
                "session_name": self.session_name,
                "timestamp": self.start_time,
                "metrics_history": self.data,
                "annotations": self.annotations
            }, f, indent=2)
        return str(filepath)

recorder = SessionRecorder()
simulation_running = False
oracle_mode = False

# Oracle Vocabulary: Semantic symbols for the field to "speak"
ORACLE_VOCAB = ["🜁", "🜂", "🜃", "🜄", "🜅", "🜆", "🜇", "🜈", "🜉", "🜊", "🜋", "🜌", "🜍", "🜎", "🜏", 
                "⦿", "⦿", "⦾", "⦿", "◎", "◌", "◍", "◎", "❂", "✺", "✹", "✷", "✸", "✹", "✺"]

async def run_simulation():
    global simulation_running
    while simulation_running:
        # Generate random input distribution
        x = torch.randn(1, 4, 128)
        x = F.softmax(x, dim=-1)
        
        # Step through reactor
        outputs, mass = reactor.react(x)
        
        # Calculate entropy for the output (Observational)
        probs = F.softmax(outputs / settings.get("temperature", 1.0), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        
        # Get latest trajectory metrics
        phase_coh = 0.0
        traj = reactor.get_trajectory()
        if traj:
            last_step = traj[-1]
            phase_coh = last_step.phase

        # The Oracle's Voice: Direct Mapping of REAL-TIME Metrics
        sampled_token = None
        if oracle_mode:
            index = int((entropy * 7 + mass * 13 + phase_coh * 17) % len(ORACLE_VOCAB))
            sampled_token = ORACLE_VOCAB[index]

        metrics = {
            "entropy": float(entropy),
            "mass": float(mass),
            "phase_coherence": float(phase_coh),
            "drift": float(last_step.deficit if traj and last_step.drift_applied else 0.0),
            "layer": int(last_step.layer_idx if traj else 0),
            "sampled_token": sampled_token,
            "oracle_mode": oracle_mode
        }
            
        recorder.record_step(metrics)
        await manager.broadcast(json.dumps(metrics))
        await asyncio.sleep(0.5)

@app.post("/record/start")
async def start_record(session_name: str = "default"):
    recorder.start(session_name)
    return {"status": "recording_started", "session": session_name}

@app.post("/record/stop")
async def stop_record():
    path = recorder.stop()
    return {"status": "recording_stopped", "file": path}

@app.post("/annotate")
async def add_annotation(note: str, marker_type: str = "manual"):
    recorder.annotate(note, marker_type)
    return {"status": "annotation_added"}

settings = {"temperature": 1.0, "coupling": 2.0}

@app.post("/settings")
async def update_settings(temperature: float = None, coupling: float = None):
    if temperature is not None: settings["temperature"] = temperature
    if coupling is not None: settings["coupling"] = coupling
    return {"status": "settings_updated", "settings": settings}

@app.post("/oracle/toggle")
async def toggle_oracle(enabled: bool):
    global oracle_mode
    oracle_mode = enabled
    return {"status": "oracle_mode_updated", "enabled": oracle_mode}

@app.get("/")
async def get():
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            global simulation_running
            if data == "start":
                if not simulation_running:
                    simulation_running = True
                    asyncio.create_task(run_simulation())
            elif data == "stop":
                simulation_running = False
            elif data == "reset":
                simulation_running = False
                reactor.reset_history()
                await manager.broadcast(json.dumps({"type": "reset"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "3.1",
        "reactor_params": reactor.hidden_dim,
        "simulation_running": simulation_running,
        "oracle_mode": oracle_mode
    }

@app.get("/config")
async def get_config():
    """Return current configuration."""
    cfg = load_config()
    return cfg if cfg else {"error": "No config file found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
