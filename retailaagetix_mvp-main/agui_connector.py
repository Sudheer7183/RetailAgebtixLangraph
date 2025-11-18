
# import json
# import uuid
# import time
# import asyncio
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio
# from contextlib import asynccontextmanager
# # Import the agents directly
# # from ingestion_agent import IngestionAgent
# from analysis_agent import AnalysisAgent
# from prediction_agent import PredictionAgent
# from decision_agent import DecisionAgent
# from execution_agent import ExecutionAgent

# from data_ingestion_agent import DataIngestionAgent

# app = FastAPI()

# # Simple in-memory store for thread states (used by /agui/run/threads/{id}/state)
# THREAD_STATES = {}

# # Enable permissive CORS for local development (adjust origins in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def _safe_emit(queue: asyncio.Queue, run_id: str, ev: dict):
#     """Place event into asyncio queue with proper error handling."""
#     ev.setdefault("run_id", run_id)
    
#     try:
#         # Get the current event loop
#         loop = asyncio.get_running_loop()
#         # Use call_soon_threadsafe to safely add to queue
#         loop.call_soon_threadsafe(queue.put_nowait, ev)
#     except RuntimeError:
#         # If no loop is running, create a new one
#         try:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             queue.put_nowait(ev)
#         except Exception as e:
#             print(f"Failed to emit event: {e}")


# def _run_pipeline_sync(queue: asyncio.Queue, run_id: str, input_state: dict):
#     """Synchronous runner with orchestrator-aligned event emission."""
#     try:
#         state = input_state or {}
#         category = input_state.get("category", "perishables").lower()

#         _safe_emit(queue, run_id, {"type": "RUN_STARTED", "timestamp": time.time()})
#         print(f"[{run_id}] RUN_STARTED (category: {category})")

#         # ───────────────────────────────
#         # INGESTION
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "ingestion"})
#         print(f"[{run_id}] STEP_STARTED: ingestion")

#         ingestion_agent = DataIngestionAgent()
#         ingestion_output = ingestion_agent.run()
#         state["ingestion"] = ingestion_output

#         # Select the appropriate subset
#         if category == "bakery":
#             print(f"[{run_id}] 🧁 Using Bakery data subset for analysis.")
#             ingestion_selected = {
#                 "sales_history": ingestion_output["bakery"]["sales_history"],
#                 "stock": ingestion_output["bakery"]["stock"],
#                 "weather": ingestion_output["weather"],
#                 "calendar_events": ingestion_output["calendar_events"],
#                 "footfall_current": ingestion_output.get("footfall_current", {}),
#                 "footfall_history": ingestion_output.get("footfall_history", {}),
#                 "user_notes": ["Bakery analysis focus"],
#                 "weather_forecast": ingestion_output["weather_forecast"],
#                 "historical_inventory": ingestion_output["bakery"]["historical_inventory"]
#             }
#         else:
#             print(f"[{run_id}] 🥦 Using Perishables data subset for analysis.")
#             ingestion_selected = ingestion_output  # normal data

#         _safe_emit(queue, run_id, {
#             "type": "STEP_FINISHED",
#             "step": "ingestion",
#             "data": ingestion_selected
#         })
#         time.sleep(0.1)

#         # ───────────────────────────────
#         # ANALYSIS
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "analysis"})
#         print(f"[{run_id}] STEP_STARTED: analysis")

#         analysis_agent = AnalysisAgent()
#         analysis_output = analysis_agent.run(ingestion_selected)
#         state["analysis"] = analysis_output

#         _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "analysis", "data": analysis_output})
#         time.sleep(0.1)

#         # ───────────────────────────────
#         # PREDICTION TRAINING
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "prediction_training"})
#         print(f"[{run_id}] STEP_STARTED: prediction_training")

#         prediction_agent = PredictionAgent()
#         for item, hist in ingestion_selected["sales_history"].items():
#             prediction_agent.train_item_model(item, hist, ingestion_selected["weather"]["temperature"])

#         _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "prediction_training"})
#         time.sleep(0.1)

#         # ───────────────────────────────
#         # PREDICTION
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "prediction"})
#         print(f"[{run_id}] STEP_STARTED: prediction")

#         prediction_output = prediction_agent.run(analysis_output)
#         state["prediction"] = prediction_output

#         _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "prediction", "data": prediction_output})
#         time.sleep(0.1)

#         # ───────────────────────────────
#         # DECISION
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "decision"})
#         print(f"[{run_id}] STEP_STARTED: decision")

#         decision_agent = DecisionAgent()
#         decision_output = decision_agent.run(analysis_output, prediction_output)
#         state["decision"] = decision_output

#         _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "decision", "data": decision_output})
#         time.sleep(0.1)

#         # ───────────────────────────────
#         # EXECUTION
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "execution"})
#         print(f"[{run_id}] STEP_STARTED: execution")

#         execution_agent = ExecutionAgent()
#         execution_output = execution_agent.run(
#             decision_output,
#             analysis_output=analysis_output,
#             prediction_output=prediction_output
#         )
#         state["execution"] = execution_output

#         _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "execution", "data": execution_output})
#         time.sleep(0.1)

#         # ───────────────────────────────
#         # FINALIZATION
#         # ───────────────────────────────
#         _safe_emit(queue, run_id, {"type": "STATE_SNAPSHOT", "state": state})
#         _safe_emit(queue, run_id, {"type": "RUN_FINISHED", "result": state})
#         print(f"[{run_id}] RUN_FINISHED")
#         return state

#     except Exception as e:
#         _safe_emit(queue, run_id, {"type": "RUN_ERROR", "error": str(e)})
#         print(f"[{run_id}] RUN_ERROR: {e}")
#         raise


# @app.post("/agui/run")
# async def run_handler(request: Request):
#     """
#     Accepts a JSON body (optional initial state) and streams AG-UI events as SSE.
#     """
#     try:
#         try:
#             payload = await request.json()
#         except Exception:
#             payload = {}
        
#         run_id = str(uuid.uuid4())
#         q: asyncio.Queue = asyncio.Queue()

#         loop = asyncio.get_event_loop()
#         # Run pipeline in executor so it doesn't block asyncio loop
#         run_future = loop.run_in_executor(None, _run_pipeline_sync, q, run_id, payload)

#         async def event_generator():
#             """Generate SSE events with proper formatting and error handling"""
#             try:
#                 # Send initial connection event
#                 yield "data: " + json.dumps({
#                     "type": "CONNECTION_ESTABLISHED", 
#                     "run_id": run_id,
#                     "timestamp": time.time()
#                 }, default=str) + "\n\n"
                
#                 # Process events from queue
#                 while True:
#                     try:
#                         # Wait for event with timeout
#                         ev = await asyncio.wait_for(q.get(), timeout=1.0)
                        
#                         # Store state snapshots
#                         try:
#                             ev_type = ev.get("type")
#                             if ev_type == "STATE_SNAPSHOT":
#                                 THREAD_STATES[run_id] = ev.get("state")
#                             elif ev_type == "RUN_FINISHED":
#                                 THREAD_STATES[run_id] = ev.get("result")
#                         except Exception:
#                             pass
                        
#                         # Send event as SSE
#                         event_line = "data: " + json.dumps(ev, default=str) + "\n\n"
#                         yield event_line
                        
#                         # Check if we should stop
#                         if ev.get("type") in ("RUN_FINISHED", "RUN_ERROR"):
#                             break
                            
#                     except asyncio.TimeoutError:
#                         # Send heartbeat to keep connection alive
#                         yield "data: " + json.dumps({
#                             "type": "HEARTBEAT",
#                             "run_id": run_id,
#                             "timestamp": time.time()
#                         }, default=str) + "\n\n"
                        
#                         # Check if the background task is still running
#                         if run_future.done():
#                             try:
#                                 # Get the result to propagate any exceptions
#                                 await run_future
#                                 # If we get here, the task completed successfully
#                                 yield "data: " + json.dumps({
#                                     "type": "RUN_COMPLETED_EXTERNALLY",
#                                     "run_id": run_id
#                                 }, default=str) + "\n\n"
#                             except Exception as e:
#                                 yield "data: " + json.dumps({
#                                     "type": "RUN_ERROR",
#                                     "run_id": run_id,
#                                     "error": str(e)
#                                 }, default=str) + "\n\n"
#                             break
                
#                 # Ensure the background task is complete
#                 if not run_future.done():
#                     try:
#                         await run_future
#                     except Exception as e:
#                         yield "data: " + json.dumps({
#                             "type": "RUN_ERROR",
#                             "run_id": run_id,
#                             "error": str(e)
#                         }, default=str) + "\n\n"
                        
#             except asyncio.CancelledError:
#                 # Client disconnected
#                 yield "data: " + json.dumps({
#                     "type": "RUN_CANCELLED", 
#                     "run_id": run_id
#                 }, default=str) + "\n\n"
#             except Exception as e:
#                 # Unexpected error
#                 yield "data: " + json.dumps({
#                     "type": "STREAM_ERROR",
#                     "run_id": run_id,
#                     "error": str(e)
#                 }, default=str) + "\n\n"

#         return StreamingResponse(
#             event_generator(), 
#             media_type="text/event-stream",
#             headers={
#                 "Cache-Control": "no-cache",
#                 "Connection": "keep-alive",
#                 "Access-Control-Allow-Origin": "*",
#                 "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
#                 "Access-Control-Allow-Headers": "*",
#             }
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# # --- AG-UI compatibility endpoints (helpers for frontend) ---
# @app.post("/agui/run/assistants/search")
# async def assistants_search(request: Request):
#     """Return a small list of available assistants for the ai-ui frontend to discover."""
#     try:
#         # Optional: read filters from payload = await request.json()
#         payload = {}
#         try:
#             payload = await request.json()
#         except Exception:
#             payload = {}
#         # Minimal assistant description — adapt to your frontend's expectations
#         assistants = [
#             {
#                 "id": "retail-orchestrator",
#                 "name": "RetailAgentix Orchestrator",
#                 "description": "Orchestrator-based assistant exposing ingestion/analysis/prediction/decision/execution",
#                 "capabilities": ["ingestion", "analysis", "prediction", "decision", "execution"],
#             }
#         ]
#         return {"assistants": assistants}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/agui/run/threads/{thread_id}/state")
# async def thread_state(thread_id: str):
#     """Return latest stored state/result for a run/thread. Returns 404 if run not found."""
#     st = THREAD_STATES.get(thread_id)
#     if st is None:
#         raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found or not finished yet")
#     return {"run_id": thread_id, "state": st}


import json
import uuid
import time
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from IPython.display import Image, display
# Import the agents directly
from data_ingestion_agent import DataIngestionAgent
from analysis_agent import AnalysisAgent
from prediction_agent import PredictionAgent
from decision_agent import DecisionAgent
from execution_agent import ExecutionAgent

app = FastAPI()

# Simple in-memory store for thread states (used by /agui/run/threads/{id}/state)
THREAD_STATES = {}
# Store for pending human approvals
PENDING_APPROVALS = {}

# Enable permissive CORS for local development (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _safe_emit(queue: asyncio.Queue, run_id: str, ev: dict):
    """Place event into asyncio queue with proper error handling."""
    ev.setdefault("run_id", run_id)
    
    try:
        # Get the current event loop
        loop = asyncio.get_running_loop()
        # Use call_soon_threadsafe to safely add to queue
        loop.call_soon_threadsafe(queue.put_nowait, ev)
    except RuntimeError:
        # If no loop is running, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            queue.put_nowait(ev)
        except Exception as e:
            print(f"Failed to emit event: {e}")

class AgentState(TypedDict):
    run_id: str
    category: str
    ingestion: Dict[str, Any]
    analysis: Dict[str, Any]
    prediction: Dict[str, Any]
    decision: Dict[str, Any]
    execution: Dict[str, Any]
    prediction_agent: Any  # To hold the trained PredictionAgent instance
    human_approved_decisions: Dict[str, Any]  # Store human-modified decisions

def _run_pipeline_sync(queue: asyncio.Queue, run_id: str, input_state: dict):
    """LangGraph-based runner with human-in-the-loop at decision stage."""
    try:
        state = input_state or {}
        category = input_state.get("category", "perishables").lower()
        state.update({"run_id": run_id, "category": category})

        _safe_emit(queue, run_id, {"type": "RUN_STARTED", "timestamp": time.time()})
        print(f"[{run_id}] RUN_STARTED (category: {category})")

        # ════════════════════════════════════
        # DEFINE NODES WITH EMITS
        # ════════════════════════════════════

        def ingestion_node(state: AgentState) -> Dict[str, Any]:
            _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "ingestion"})
            print(f"[{run_id}] STEP_STARTED: ingestion")

            ingestion_agent = DataIngestionAgent()
            ingestion_output = ingestion_agent.run()
            state["ingestion"] = ingestion_output

            # Select the appropriate subset
            if category == "bakery":
                print(f"[{run_id}] 🧁 Using Bakery data subset for analysis.")
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
                print(f"[{run_id}] 🥦 Using Perishables data subset for analysis.")
                ingestion_selected = ingestion_output  # normal data

            _safe_emit(queue, run_id, {
                "type": "STEP_FINISHED",
                "step": "ingestion",
                "data": ingestion_selected
            })
            time.sleep(0.1)

            return {"ingestion": ingestion_selected}

        def analysis_node(state: AgentState) -> Dict[str, Any]:
            _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "analysis"})
            print(f"[{run_id}] STEP_STARTED: analysis")

            analysis_agent = AnalysisAgent()
            analysis_output = analysis_agent.run(state["ingestion"])
            state["analysis"] = analysis_output

            _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "analysis", "data": analysis_output})
            time.sleep(0.1)

            return {"analysis": analysis_output}

        def prediction_training_node(state: AgentState) -> Dict[str, Any]:
            _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "prediction_training"})
            print(f"[{run_id}] STEP_STARTED: prediction_training")

            prediction_agent = PredictionAgent()
            for item, hist in state["ingestion"]["sales_history"].items():
                prediction_agent.train_item_model(item, hist, state["ingestion"]["weather"]["temperature"])

            _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "prediction_training"})
            time.sleep(0.1)

            return {"prediction_agent": prediction_agent}  # Store trained agent in state

        def prediction_node(state: AgentState) -> Dict[str, Any]:
            _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "prediction"})
            print(f"[{run_id}] STEP_STARTED: prediction")

            prediction_agent = state["prediction_agent"]
            prediction_output = prediction_agent.run(state["analysis"])
            state["prediction"] = prediction_output

            _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "prediction", "data": prediction_output})
            time.sleep(0.1)

            return {"prediction": prediction_output}

        def decision_node(state: AgentState) -> Dict[str, Any]:
            _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "decision"})
            print(f"[{run_id}] STEP_STARTED: decision")

            decision_agent = DecisionAgent()
            decision_output = decision_agent.run(state["analysis"], state["prediction"])
            state["decision"] = decision_output

            _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "decision", "data": decision_output})
            time.sleep(0.1)

            # HUMAN-IN-THE-LOOP: Wait for approval
            _safe_emit(queue, run_id, {
                "type": "HUMAN_APPROVAL_REQUIRED",
                "step": "decision",
                "data": decision_output,
                "message": "Please review and approve/modify the pricing decisions"
            })
            print(f"[{run_id}] Waiting for human approval...")

            # Store the pending approval request
            approval_event = asyncio.Event()
            PENDING_APPROVALS[run_id] = {
                "event": approval_event,
                "original_decisions": decision_output,
                "approved_decisions": None
            }

            # Wait for approval (blocking wait with timeout)
            timeout = 3600  # 1 hour timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                if PENDING_APPROVALS[run_id].get("approved_decisions") is not None:
                    approved_decisions = PENDING_APPROVALS[run_id]["approved_decisions"]
                    print(f"[{run_id}] Human approval received")
                    
                    # Emit approval received event
                    _safe_emit(queue, run_id, {
                        "type": "HUMAN_APPROVAL_RECEIVED",
                        "step": "decision",
                        "data": approved_decisions
                    })
                    
                    # Clean up
                    del PENDING_APPROVALS[run_id]
                    
                    return {"decision": approved_decisions, "human_approved_decisions": approved_decisions}
                
                time.sleep(0.5)  # Check every 500ms

            # Timeout - use original decisions
            print(f"[{run_id}] Human approval timeout - using original decisions")
            _safe_emit(queue, run_id, {
                "type": "HUMAN_APPROVAL_TIMEOUT",
                "step": "decision"
            })
            del PENDING_APPROVALS[run_id]
            
            return {"decision": decision_output}

        def execution_node(state: AgentState) -> Dict[str, Any]:
            _safe_emit(queue, run_id, {"type": "STEP_STARTED", "step": "execution"})
            print(f"[{run_id}] STEP_STARTED: execution")

            # Use human-approved decisions if available
            # decisions_to_execute = state.get("human_approved_decisions", state["decision"])

            if "human_approved_decisions" in state and state["human_approved_decisions"]:
                decisions_to_execute = {
                    "decisions": state["human_approved_decisions"]
                }
            else:
                decisions_to_execute = state["decision"]

            execution_agent = ExecutionAgent()
            execution_output = execution_agent.run(
                decisions_to_execute,
                analysis_output=state["analysis"],
                prediction_output=state["prediction"]
            )
            state["execution"] = execution_output

            _safe_emit(queue, run_id, {"type": "STEP_FINISHED", "step": "execution", "data": execution_output})
            time.sleep(0.1)

            return {"execution": execution_output}

        # ════════════════════════════════════
        # BUILD AND RUN GRAPH
        # ════════════════════════════════════
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

        app_graph = workflow.compile()
        try:
            # Try PNG visualization first
            display(Image(app_graph.get_graph().draw_mermaid_png()))
        except Exception:
            # Fallback to ASCII
            print(app_graph.get_graph().draw_ascii())

        final_state = app_graph.invoke(state)

        # ════════════════════════════════════
        # FINALIZATION
        # ════════════════════════════════════
        _safe_emit(queue, run_id, {"type": "STATE_SNAPSHOT", "state": final_state})
        _safe_emit(queue, run_id, {"type": "RUN_FINISHED", "result": final_state})

        # ✅ Ensure final state is stored even if no SSE client is active
        # THREAD_STATES[run_id] = final_state
        THREAD_STATES[run_id] = prune_unserializable(final_state)

        print(f"[{run_id}] RUN_FINISHED and state stored.")
        return final_state
    except Exception as e:
        _safe_emit(queue, run_id, {"type": "RUN_ERROR", "error": str(e)})
        print(f"[{run_id}] RUN_ERROR: {e}")
        raise


@app.post("/agui/run")
async def run_handler(request: Request):
    """
    Accepts a JSON body (optional initial state) and streams AG-UI events as SSE.
    """
    try:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        
        run_id = str(uuid.uuid4())
        q: asyncio.Queue = asyncio.Queue()

        loop = asyncio.get_event_loop()
        # Run pipeline in executor so it doesn't block asyncio loop
        run_future = loop.run_in_executor(None, _run_pipeline_sync, q, run_id, payload)

        async def event_generator():
            """Generate SSE events with proper formatting and error handling"""
            try:
                # Send initial connection event
                yield "data: " + json.dumps({
                    "type": "CONNECTION_ESTABLISHED", 
                    "run_id": run_id,
                    "timestamp": time.time()
                }, default=str) + "\n\n"
                
                # Process events from queue
                while True:
                    try:
                        # Wait for event with timeout
                        ev = await asyncio.wait_for(q.get(), timeout=1.0)
                        
                        # Store state snapshots
                        try:
                            ev_type = ev.get("type")
                            if ev_type == "STATE_SNAPSHOT":
                                THREAD_STATES[run_id] = ev.get("state")
                            elif ev_type == "RUN_FINISHED":
                                THREAD_STATES[run_id] = ev.get("result")
                        except Exception:
                            pass
                        
                        # Send event as SSE
                        event_line = "data: " + json.dumps(ev, default=str) + "\n\n"
                        yield event_line
                        
                        # Check if we should stop
                        if ev.get("type") in ("RUN_FINISHED", "RUN_ERROR"):
                            break
                            
                    except asyncio.TimeoutError:
                        # Send heartbeat to keep connection alive
                        yield "data: " + json.dumps({
                            "type": "HEARTBEAT",
                            "run_id": run_id,
                            "timestamp": time.time()
                        }, default=str) + "\n\n"
                        
                        # Check if the background task is still running
                        if run_future.done():
                            try:
                                # Get the result to propagate any exceptions
                                await run_future
                                # If we get here, the task completed successfully
                                yield "data: " + json.dumps({
                                    "type": "RUN_COMPLETED_EXTERNALLY",
                                    "run_id": run_id
                                }, default=str) + "\n\n"
                            except Exception as e:
                                yield "data: " + json.dumps({
                                    "type": "RUN_ERROR",
                                    "run_id": run_id,
                                    "error": str(e)
                                }, default=str) + "\n\n"
                            break
                
                # Ensure the background task is complete
                if not run_future.done():
                    try:
                        await run_future
                    except Exception as e:
                        yield "data: " + json.dumps({
                            "type": "RUN_ERROR",
                            "run_id": run_id,
                            "error": str(e)
                        }, default=str) + "\n\n"
                        
            except asyncio.CancelledError:
                # Client disconnected
                yield "data: " + json.dumps({
                    "type": "RUN_CANCELLED", 
                    "run_id": run_id
                }, default=str) + "\n\n"
            except Exception as e:
                # Unexpected error
                yield "data: " + json.dumps({
                    "type": "STREAM_ERROR",
                    "run_id": run_id,
                    "error": str(e)
                }, default=str) + "\n\n"

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agui/run/{run_id}/approve")
async def approve_decisions(run_id: str, request: Request):
    """
    Endpoint to receive human-approved decisions.
    Accepts either {"decisions": {...}} or {"decisions": {"decisions": {...}}}.
    """
    try:
        payload = await request.json()
        approved_decisions = payload.get("decisions")

        if isinstance(approved_decisions, dict) and "decisions" in approved_decisions:
            approved_decisions = approved_decisions.get("decisions")

        if run_id not in PENDING_APPROVALS:
            raise HTTPException(status_code=404, detail=f"No pending approval for run {run_id}")

        PENDING_APPROVALS[run_id]["approved_decisions"] = approved_decisions

        # Wake up any waiting event
        ev = PENDING_APPROVALS[run_id].get("event")
        if isinstance(ev, asyncio.Event):
            ev.set()

        print(f"[{run_id}] ✅ Human approval received and stored")
        return {"status": "success", "message": "Decisions approved and pipeline will continue", "run_id": run_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- AG-UI compatibility endpoints (helpers for frontend) ---
@app.post("/agui/run/assistants/search")
async def assistants_search(request: Request):
    """Return a small list of available assistants for the ai-ui frontend to discover."""
    try:
        # Optional: read filters from payload = await request.json()
        payload = {}
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        # Minimal assistant description – adapt to your frontend's expectations
        assistants = [
            {
                "id": "retail-orchestrator",
                "name": "RetailAgentix Orchestrator",
                "description": "Orchestrator-based assistant exposing ingestion/analysis/prediction/decision/execution",
                "capabilities": ["ingestion", "analysis", "prediction", "decision", "execution"],
            }
        ]
        return {"assistants": assistants}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np

def prune_unserializable(obj):
    import numpy as np
    import pandas as pd
    import types
    from sklearn.base import BaseEstimator

    # Basic types
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
    if isinstance(obj, list):
        return [prune_unserializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(prune_unserializable(v) for v in obj)

    # Skip sklearn / model / function / module objects
    if isinstance(obj, BaseEstimator) or "sklearn" in str(type(obj)) or callable(obj) or isinstance(obj, types.ModuleType):
        return f"<unserializable {type(obj).__name__}>"

    # Fallback
    try:
        return str(obj)
    except Exception:
        return f"<unserializable {type(obj).__name__}>"




from fastapi.responses import JSONResponse

@app.get("/agui/run/threads/{thread_id}/state")
async def thread_state(thread_id: str):
    st = THREAD_STATES.get(thread_id)
    if st is None:
        raise HTTPException(status_code=404, detail=f"No stored state for run {thread_id}")
    return JSONResponse(content={"run_id": thread_id, "state": st})
