
import json
import uuid
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, TypedDict, Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangGraph imports
from langgraph.graph import StateGraph, END

# Import the agents
from data_ingestion_agent import DataIngestionAgent
from analysis_agent import AnalysisAgent
from prediction_agent import PredictionAgent
from decision_agent import DecisionAgent
from execution_agent import ExecutionAgent

app = FastAPI(title="RetailAgentix AG-UI Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from ag_ui.core import (
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageChunkEvent,
    StateSnapshotEvent,
    RunErrorEvent
)
from ag_ui.encoder import EventEncoder

# ============================================================================
# DATA MODELS
# ============================================================================

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class Thread(BaseModel):
    id: str
    assistant_id: str
    created_at: float
    metadata: Dict[str, Any] = {}
    messages: List[Message] = []
    state: Dict[str, Any] = {}
    status: str = "active"

class Assistant(BaseModel):
    id: str
    name: str
    description: str
    capabilities: List[str]
    metadata: Dict[str, Any] = {}

# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

ASSISTANTS: Dict[str, Assistant] = {
    "retail-orchestrator": Assistant(
        id="retail-orchestrator",
        name="RetailAgentix Orchestrator",
        description="Multi-agent system for retail analytics with human-in-the-loop decision making",
        capabilities=["ingestion", "analysis", "prediction", "decision", "execution", "human_approval"],
        metadata={"version": "1.0.0"}
    )
}

THREADS: Dict[str, Thread] = {}
PENDING_APPROVALS: Dict[str, Dict[str, Any]] = {}  # Key: run_id
ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}  # Key: run_id, stores run metadata

# ============================================================================
# LANGGRAPH STATE
# ============================================================================

class AgentState(TypedDict):
    thread_id: str
    run_id: str
    category: str
    ingestion: Dict[str, Any]
    analysis: Dict[str, Any]
    prediction: Dict[str, Any]
    decision: Dict[str, Any]
    execution: Dict[str, Any]
    prediction_agent: Any
    human_approved_decisions: Dict[str, Any]
    messages: List[Dict[str, Any]]
    approval_requested: bool

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_emit(queue: asyncio.Queue, event: dict):
    """Safely emit events to the queue"""
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(queue.put_nowait, event)
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            queue.put_nowait(event)
        except Exception as e:
            print(f"Failed to emit event: {e}")

def prune_unserializable(obj):
    """Remove unserializable objects for JSON encoding"""
    import numpy as np
    import pandas as pd
    import types
    from sklearn.base import BaseEstimator

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, dict):
        return {k: prune_unserializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [prune_unserializable(v) for v in obj]
    
    if isinstance(obj, BaseEstimator) or "sklearn" in str(type(obj)) or callable(obj) or isinstance(obj, types.ModuleType):
        return f"<model:{type(obj).__name__}>"
    
    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"

def create_agui_event(event_type: str, data: Any, thread_id: str, run_id: str) -> dict:
    """Create AG-UI protocol compliant event"""
    return {
        "type": event_type,
        "thread_id": thread_id,
        "run_id": run_id,
        "timestamp": time.time(),
        "data": prune_unserializable(data)
    }

# ============================================================================
# LANGGRAPH PIPELINE
# ============================================================================

def build_pipeline_graph(queue: asyncio.Queue, thread_id: str, run_id: str, category: str):
    """Build the LangGraph pipeline with event emission"""
    
    def ingestion_node(state: AgentState) -> Dict[str, Any]:
        _safe_emit(queue, create_agui_event(
            "agent:step:start", 
            {"agent": "ingestion", "step": "data_collection"},
            thread_id, run_id
        ))
        
        ingestion_agent = DataIngestionAgent()
        ingestion_output = ingestion_agent.run()
        
        if category == "bakery":
            ingestion_selected = {
                "sales_history": ingestion_output["bakery"]["sales_history"],
                "stock": ingestion_output["bakery"]["stock"],
                "weather": ingestion_output["weather"],
                "calendar_events": ingestion_output["calendar_events"],
                "footfall_current": ingestion_output.get("footfall_current", {}),
                "footfall_history": ingestion_output.get("footfall_history", {}),
                "user_notes": ["Bakery analysis focus"],
                "weather_forecast": ingestion_output["weather_forecast"],
                "historical_inventory": ingestion_output["bakery"]["historical_inventory"]
            }
        else:
            ingestion_selected = ingestion_output
        
        _safe_emit(queue, create_agui_event(
            "agent:step:complete",
            {"agent": "ingestion", "result": ingestion_selected},
            thread_id, run_id
        ))
        
        state["messages"].append({
            "role": "assistant",
            "content": f"Data ingestion completed for {category}",
            "timestamp": time.time(),
            "metadata": {"agent": "ingestion", "items_processed": len(ingestion_selected.get("stock", {}))}
        })
        
        return {"ingestion": ingestion_selected, "messages": state["messages"]}
    
    def analysis_node(state: AgentState) -> Dict[str, Any]:
        _safe_emit(queue, create_agui_event(
            "agent:step:start",
            {"agent": "analysis", "step": "data_analysis"},
            thread_id, run_id
        ))
        
        analysis_agent = AnalysisAgent()
        analysis_output = analysis_agent.run(state["ingestion"])
        
        _safe_emit(queue, create_agui_event(
            "agent:step:complete",
            {"agent": "analysis", "result": analysis_output},
            thread_id, run_id
        ))
        
        state["messages"].append({
            "role": "assistant",
            "content": "Analysis completed - trends and spoilage risks calculated",
            "timestamp": time.time(),
            "metadata": {"agent": "analysis", "items_analyzed": len(analysis_output.get("analysis", {}))}
        })
        
        return {"analysis": analysis_output, "messages": state["messages"]}
    
    def prediction_training_node(state: AgentState) -> Dict[str, Any]:
        _safe_emit(queue, create_agui_event(
            "agent:step:start",
            {"agent": "prediction", "step": "model_training"},
            thread_id, run_id
        ))
        
        prediction_agent = PredictionAgent()
        for item, hist in state["ingestion"]["sales_history"].items():
            prediction_agent.train_item_model(
                item, hist, 
                state["ingestion"]["weather"]["temperature"]
            )
        
        _safe_emit(queue, create_agui_event(
            "agent:step:complete",
            {"agent": "prediction", "step": "model_training", "status": "trained"},
            thread_id, run_id
        ))
        
        return {"prediction_agent": prediction_agent}
    
    def prediction_node(state: AgentState) -> Dict[str, Any]:
        _safe_emit(queue, create_agui_event(
            "agent:step:start",
            {"agent": "prediction", "step": "forecasting"},
            thread_id, run_id
        ))
        
        prediction_agent = state["prediction_agent"]
        prediction_output = prediction_agent.run(state["analysis"])
        
        _safe_emit(queue, create_agui_event(
            "agent:step:complete",
            {"agent": "prediction", "result": prediction_output},
            thread_id, run_id
        ))
        
        state["messages"].append({
            "role": "assistant",
            "content": "Demand forecasting completed",
            "timestamp": time.time(),
            "metadata": {"agent": "prediction", "predictions_made": len(prediction_output.get("predictions", {}))}
        })
        
        return {"prediction": prediction_output, "messages": state["messages"]}
    
    def decision_node(state: AgentState) -> Dict[str, Any]:
        _safe_emit(queue, create_agui_event(
            "agent:step:start",
            {"agent": "decision", "step": "pricing_optimization"},
            thread_id, run_id
        ))
        
        decision_agent = DecisionAgent()
        decision_output = decision_agent.run(state["analysis"], state["prediction"])
        
        _safe_emit(queue, create_agui_event(
            "agent:step:complete",
            {"agent": "decision", "result": decision_output},
            thread_id, run_id
        ))
        
        # HUMAN APPROVAL REQUIRED
        _safe_emit(queue, create_agui_event(
            "human:approval:required",
            {
                "agent": "decision",
                "decisions": decision_output,
                "message": "Please review and approve/modify pricing decisions"
            },
            thread_id, run_id
        ))
        
        state["messages"].append({
            "role": "assistant",
            "content": "⏸️ Pricing decisions ready for human review",
            "timestamp": time.time(),
            "metadata": {
                "agent": "decision",
                "requires_approval": True,
                "decision_count": len(decision_output.get("decisions", {}))
            }
        })
        
        # ✅ CRITICAL FIX: Store approval in PENDING_APPROVALS with run_id as key
        PENDING_APPROVALS[run_id] = {
            "thread_id": thread_id,
            "run_id": run_id,
            "original_decisions": decision_output,
            "approved_decisions": None,
            "timestamp": time.time()
        }
        
        print(f"[Backend] Stored approval request for run_id: {run_id}")
        print(f"[Backend] PENDING_APPROVALS keys: {list(PENDING_APPROVALS.keys())}")
        
        # Wait for approval
        timeout = 3600
        start_time = time.time()
        poll_interval = 0.5
        
        while time.time() - start_time < timeout:
            if run_id in PENDING_APPROVALS:
                if PENDING_APPROVALS[run_id].get("approved_decisions") is not None:
                    approved_decisions = PENDING_APPROVALS[run_id]["approved_decisions"]
                    
                    print(f"[Backend] Approval received for run_id: {run_id}")
                    
                    _safe_emit(queue, create_agui_event(
                        "human:approval:received",
                        {"agent": "decision", "approved_decisions": approved_decisions},
                        thread_id, run_id
                    ))
                    
                    state["messages"].append({
                        "role": "user",
                        "content": "✅ Decisions approved by human operator",
                        "timestamp": time.time(),
                        "metadata": {"approval_status": "approved"}
                    })
                    
                    # Clean up
                    del PENDING_APPROVALS[run_id]
                    
                    return {
                        "decision": approved_decisions,
                        "human_approved_decisions": approved_decisions,
                        "messages": state["messages"],
                        "approval_requested": False
                    }
            else:
                print(f"[Backend] run_id {run_id} no longer in PENDING_APPROVALS - approval received")
                break
            
            time.sleep(poll_interval)
        
        # Timeout
        print(f"[Backend] Approval timeout for run_id: {run_id}")
        _safe_emit(queue, create_agui_event(
            "human:approval:timeout",
            {"agent": "decision"},
            thread_id, run_id
        ))
        
        if run_id in PENDING_APPROVALS:
            del PENDING_APPROVALS[run_id]
        
        return {
            "decision": decision_output,
            "messages": state["messages"],
            "approval_requested": False
        }
    
    def execution_node(state: AgentState) -> Dict[str, Any]:
        _safe_emit(queue, create_agui_event(
            "agent:step:start",
            {"agent": "execution", "step": "price_update"},
            thread_id, run_id
        ))
        
        decisions_to_execute = state.get("human_approved_decisions") or state["decision"]
        if isinstance(decisions_to_execute, dict) and "decisions" not in decisions_to_execute:
            decisions_to_execute = {"decisions": decisions_to_execute}
        
        execution_agent = ExecutionAgent()
        execution_output = execution_agent.run(
            decisions_to_execute,
            analysis_output=state["analysis"],
            prediction_output=state["prediction"]
        )
        
        _safe_emit(queue, create_agui_event(
            "agent:step:complete",
            {"agent": "execution", "result": execution_output},
            thread_id, run_id
        ))
        
        state["messages"].append({
            "role": "assistant",
            "content": "✅ Execution completed - prices updated and orders placed",
            "timestamp": time.time(),
            "metadata": {
                "agent": "execution",
                "items_updated": len(execution_output.get("execution_log", []))
            }
        })
        
        return {"execution": execution_output, "messages": state["messages"]}
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("ingestion", ingestion_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("prediction_training", prediction_training_node)
    workflow.add_node("prediction", prediction_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("execution", execution_node)
    
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", "analysis")
    workflow.add_edge("analysis", "prediction_training")
    workflow.add_edge("prediction_training", "prediction")
    workflow.add_edge("prediction", "decision")
    workflow.add_edge("decision", "execution")
    workflow.add_edge("execution", END)
    
    return workflow.compile()

# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def _run_pipeline_sync(queue: asyncio.Queue, thread_id: str, run_id: str, input_state: dict):
    """Run the pipeline synchronously with event emission"""
    try:
        category = input_state.get("category", "perishables").lower()
        
        # Register active run
        ACTIVE_RUNS[run_id] = {
            "thread_id": thread_id,
            "run_id": run_id,
            "status": "running",
            "started_at": time.time()
        }
        
        _safe_emit(queue, create_agui_event(
            "run:start",
            {"category": category, "input": input_state},
            thread_id, run_id
        ))
        
        initial_state = {
            "thread_id": thread_id,
            "run_id": run_id,
            "category": category,
            "messages": [],
            "approval_requested": False
        }
        
        app_graph = build_pipeline_graph(queue, thread_id, run_id, category)
        final_state = app_graph.invoke(initial_state)
        
        # Update thread
        if thread_id in THREADS:
            THREADS[thread_id].state = prune_unserializable(final_state)
            THREADS[thread_id].messages = final_state.get("messages", [])
            THREADS[thread_id].status = "completed"
        
        # Update active run
        ACTIVE_RUNS[run_id]["status"] = "completed"
        ACTIVE_RUNS[run_id]["completed_at"] = time.time()
        
        _safe_emit(queue, create_agui_event(
            "run:complete",
            {"final_state": prune_unserializable(final_state)},
            thread_id, run_id
        ))
        
        return final_state
        
    except Exception as e:
        print(f"[Backend] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        
        _safe_emit(queue, create_agui_event(
            "run:error",
            {"error": str(e)},
            thread_id, run_id
        ))
        
        if thread_id in THREADS:
            THREADS[thread_id].status = "error"
        
        if run_id in ACTIVE_RUNS:
            ACTIVE_RUNS[run_id]["status"] = "error"
            ACTIVE_RUNS[run_id]["error"] = str(e)
        
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/assistants")
async def list_assistants():
    """List all available assistants"""
    return {"assistants": list(ASSISTANTS.values())}

@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant details"""
    if assistant_id not in ASSISTANTS:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return ASSISTANTS[assistant_id]

@app.post("/threads")
async def create_thread(request: Request):
    """Create a new thread"""
    try:
        payload = await request.json()
    except:
        payload = {}
    
    thread_id = str(uuid.uuid4())
    thread = Thread(
        id=thread_id,
        assistant_id=payload.get("assistant_id", "retail-orchestrator"),
        created_at=time.time(),
        metadata=payload.get("metadata", {})
    )
    
    THREADS[thread_id] = thread
    return thread

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get thread details"""
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")
    return THREADS[thread_id]

@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    """Get thread state"""
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "state": THREADS[thread_id].state}

@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request):
    """Create and execute a new run on a thread (with SSE streaming)"""
    
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    try:
        payload = await request.json()
    except:
        payload = {}
    
    run_id = str(uuid.uuid4())
    input_data = payload.get("input", {})
    
    print(f"[Backend] Starting run_id: {run_id} for thread_id: {thread_id}")
    
    THREADS[thread_id].status = "running"
    
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    run_future = loop.run_in_executor(None, _run_pipeline_sync, q, thread_id, run_id, input_data)

    async def event_generator():
        """Generate SSE events with proper state snapshots"""
        encoder = EventEncoder(accept=request.headers.get("accept"))
        message_id = str(uuid.uuid4())
        accumulated_state = {}

        try:
            # ✅ Send run_started with explicit run_id
            run_started = RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=thread_id,
                run_id=run_id
            )
            
            print(f"[Backend] Sending RUN_STARTED - thread: {thread_id}, run: {run_id}")
            yield encoder.encode(run_started)
            
            # ✅ Also send run_id as a text chunk to ensure it's captured
            yield encoder.encode(TextMessageChunkEvent(
                type=EventType.TEXT_MESSAGE_CHUNK,
                message_id=message_id,
                delta=f"[RUN_ID:{run_id}]\n"
            ))

            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=1.0)
                    event_type = ev.get("type")
                    event_data = ev.get("data", {})

                    if event_type == "agent:step:start":
                        agent_name = event_data.get('agent', 'unknown')
                        yield encoder.encode(TextMessageChunkEvent(
                            type=EventType.TEXT_MESSAGE_CHUNK,
                            message_id=message_id,
                            delta=f"🔄 {agent_name} starting...\n"
                        ))

                    elif event_type == "agent:step:complete":
                        agent_name = event_data.get('agent', 'unknown')
                        agent_result = event_data.get('result', {})
                        accumulated_state[agent_name] = agent_result
                        
                        yield encoder.encode(TextMessageChunkEvent(
                            type=EventType.TEXT_MESSAGE_CHUNK,
                            message_id=message_id,
                            delta=f"✅ {agent_name} complete\n"
                        ))
                        
                        pruned_state = prune_unserializable(accumulated_state)
                        pruned_state["_run_id"] = run_id
                        yield encoder.encode(StateSnapshotEvent(
                            type=EventType.STATE_SNAPSHOT,
                            snapshot=pruned_state
                        ))

                    elif event_type == "human:approval:required":
                        decisions = event_data.get('decisions', {})
                        if decisions:
                            accumulated_state['decision'] = decisions
                        
                        yield encoder.encode(TextMessageChunkEvent(
                            type=EventType.TEXT_MESSAGE_CHUNK,
                            message_id=message_id,
                            delta=f"✋ Human approval required - please review decisions\n"
                        ))
                        
                        pruned_state = prune_unserializable(accumulated_state)
                        pruned_state["_run_id"] = run_id
                        yield encoder.encode(StateSnapshotEvent(
                            type=EventType.STATE_SNAPSHOT,
                            snapshot=pruned_state
                        ))

                    elif event_type == "human:approval:received":
                        approved = event_data.get('approved_decisions', {})
                        accumulated_state['human_approved_decisions'] = approved
                        
                        yield encoder.encode(TextMessageChunkEvent(
                            type=EventType.TEXT_MESSAGE_CHUNK,
                            message_id=message_id,
                            delta=f"✅ Approval received - continuing execution\n"
                        ))

                    elif event_type == "run:complete":
                        final_state_data = event_data.get('final_state', {})
                        accumulated_state.update(final_state_data)
                        
                        pruned_state = prune_unserializable(accumulated_state)
                        pruned_state["_run_id"] = run_id
                        yield encoder.encode(StateSnapshotEvent(
                            type=EventType.STATE_SNAPSHOT,
                            snapshot=pruned_state
                        ))
                        
                        yield encoder.encode(RunFinishedEvent(
                            type=EventType.RUN_FINISHED,
                            thread_id=thread_id,
                            run_id=run_id
                        ))
                        break

                    elif event_type == "run:error":
                        error_msg = event_data.get('error', 'Unknown error')
                        yield encoder.encode(RunErrorEvent(
                            type=EventType.RUN_ERROR,
                            message=error_msg
                        ))
                        break

                except asyncio.TimeoutError:
                    continue

            if not run_future.done():
                await run_future

        except Exception as e:
            print(f"[ERROR] Event generator failed: {e}")
            import traceback
            traceback.print_exc()
            
            yield encoder.encode(RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(e)
            ))
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/threads/{thread_id}/runs/{run_id}/approve")
async def approve_run_decisions(thread_id: str, run_id: str, request: Request):
    """Submit human approval for pending decisions"""
    
    print(f"[Backend] Approval request received for thread_id: {thread_id}, run_id: {run_id}")
    print(f"[Backend] Current PENDING_APPROVALS keys: {list(PENDING_APPROVALS.keys())}")
    
    if run_id not in PENDING_APPROVALS:
        # Additional debug info
        print(f"[Backend] ACTIVE_RUNS: {list(ACTIVE_RUNS.keys())}")
        raise HTTPException(
            status_code=404, 
            detail=f"No pending approval for run_id: {run_id}. Available: {list(PENDING_APPROVALS.keys())}"
        )
    
    try:
        payload = await request.json()
        approved_decisions = payload.get("decisions")
        
        if isinstance(approved_decisions, dict) and "decisions" in approved_decisions:
            approved_decisions = approved_decisions["decisions"]
        
        print(f"[Backend] Setting approved_decisions for run_id: {run_id}")
        PENDING_APPROVALS[run_id]["approved_decisions"] = approved_decisions
        
        return {
            "status": "success",
            "message": "Decisions approved, pipeline will continue",
            "thread_id": thread_id,
            "run_id": run_id
        }
        
    except Exception as e:
        print(f"[Backend] Approval error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint
@app.get("/debug/pending-approvals")
async def debug_pending_approvals():
    """Debug endpoint to see pending approvals"""
    return {
        "pending_approvals": list(PENDING_APPROVALS.keys()),
        "active_runs": list(ACTIVE_RUNS.keys()),
        "threads": list(THREADS.keys())
    }

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

@app.post("/agui/run")
async def legacy_run_handler(request: Request):
    """Legacy endpoint for backward compatibility"""
    try:
        payload = await request.json()
    except:
        payload = {}
    
    thread = Thread(
        id=str(uuid.uuid4()),
        assistant_id="retail-orchestrator",
        created_at=time.time()
    )
    THREADS[thread.id] = thread
    
    return await create_run(thread.id, request)

@app.post("/agui/run/{run_id}/approve")
async def legacy_approve(run_id: str, request: Request):
    """Legacy approval endpoint"""
    thread_id = None
    
    if run_id in PENDING_APPROVALS:
        thread_id = PENDING_APPROVALS[run_id].get("thread_id")
    
    if not thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return await approve_run_decisions(thread_id, run_id, request)

@app.post("/agui/run/assistants/search")
async def legacy_assistants_search():
    """Legacy assistant search"""
    return await list_assistants()

@app.get("/agui/run/threads/{thread_id}/state")
async def legacy_thread_state(thread_id: str):
    """Legacy thread state endpoint"""
    return await get_thread_state(thread_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)