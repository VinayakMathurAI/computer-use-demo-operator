from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import List, Dict, Any
from anthropic import Anthropic
import os
from pathlib import PosixPath

from computer_use_demo.loop import sampling_loop, APIProvider, PROVIDER_TO_DEFAULT_MODEL_NAME
from computer_use_demo.tools import ToolResult

# Add these constants from streamlit.py
CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

# Store active WebSocket connections
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("\n=== New WebSocket Connection ===")
    await websocket.accept()
    print("WebSocket connection accepted")
    active_connections.append(websocket)
    
    # Initialize conversation state
    conversation_messages = []
    
    try:
        while True:
            print("\nWaiting for message...")
            message = await websocket.receive_text()
            print(f"=== Received Message from Client ===")
            print(f"Raw message: {message}")
            data = json.loads(message)
            print(f"Parsed data: {json.dumps(data, indent=2)}")
            print(f"Current conversation length: {len(conversation_messages)}")
            print("================================")
            
            # Callback to send Claude's responses to frontend
            async def output_callback(content: Dict[str, Any]):
                print(f"\n=== Sending to Client (output_callback) ===")
                print(f"Content: {json.dumps(content, indent=2)}")
                await websocket.send_json({
                    "type": "message",
                    "content": content
                })
                print("=========================================")
            
            # Callback to send tool outputs to frontend
            async def tool_output_callback(result: ToolResult, tool_id: str):
                print(f"\n=== Sending Tool Output to Client ===")
                print(f"Result: {result}")
                print(f"Tool ID: {tool_id}")
                await websocket.send_json({
                    "type": "tool_output",
                    "content": {
                        "output": result.output,
                        "error": result.error,
                        "base64_image": result.base64_image
                    }
                })
                print("===================================")

            # Add this callback definition
            async def api_response_callback(request, response, error):
                print("\n=== API Response Callback ===")
                if error:
                    print(f"API Error: {error}")
                    await websocket.send_json({
                        "type": "error",
                        "content": str(error)
                    })
                else:
                    print("API Response received successfully")
                print("=============================")

            # Get API key and provider like streamlit.py does
            api_key = load_from_storage("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
            provider = os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
            model = PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider(provider)]

            print(f"\n=== Calling Claude ===")
            print(f"Provider: {provider}")
            print(f"Model: {model}")
            print(f"API Key present: {bool(api_key)}")
            print(f"Sending conversation with {len(conversation_messages)} previous messages")
            
            try:
                # Add new user message to conversation
                conversation_messages.append({
                    "role": "user",
                    "content": data["message"]
                })
                
                # Pass the full conversation history to sampling_loop
                messages = await sampling_loop(
                    model=model,
                    provider=APIProvider(provider),
                    system_prompt_suffix="",
                    messages=conversation_messages,  # Using full conversation history
                    output_callback=output_callback,
                    tool_output_callback=tool_output_callback,
                    api_response_callback=api_response_callback,
                    api_key=api_key,
                )
                
                # Update conversation history with Claude's response
                conversation_messages = messages
                
                print(f"\n=== Claude Response Complete ===")
                print(f"Updated conversation length: {len(conversation_messages)}")
                print(f"Last message role: {conversation_messages[-1]['role']}")
            except Exception as e:
                print(f"\n!!! Error in sampling loop !!!: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
            
    except Exception as e:
        print(f"\n!!! WebSocket Error !!!: {str(e)}")
    finally:
        active_connections.remove(websocket)
        print("\n=== WebSocket connection closed ===")


@app.on_event("shutdown")
async def shutdown_event():
    # Close all WebSocket connections on shutdown
    for connection in active_connections:
        await connection.close()


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    
    # Add this debug log
    api_key = load_from_storage("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
    print(f"API Key {'found' if api_key else 'not found'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8501)