import streamlit as st
from typing import TypedDict, List, Union, Dict
from langgraph.graph import StateGraph, END
import boto3
import json
from botocore.config import Config
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# AWS Configuration
def create_bedrock_client():
    config = Config(
        retries={'max_attempts': 10, 'mode': 'standard'},
        max_pool_connections=200
    )
    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-west-2'),
        config=config
    )

bedrock = create_bedrock_client()

def invoke_claude(prompt: str) -> str:
    """Invoke Claude with error handling."""
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        logger.error(f"Error invoking Claude: {e}")
        return f"Error: Unable to process request. {str(e)}"

class AgentState(TypedDict):
    """State definition for the WiFi troubleshooting workflow."""
    messages: List[dict]
    current_step: str
    issue_identified: bool
    diagnostics_complete: bool
    user_approval: bool
    resolution_complete: bool

def identify_issue(state: AgentState) -> AgentState:
    """Initial issue identification step."""
    logger.info("Identifying issue...")
    
    if not state["messages"]:
        # Initial greeting
        state["messages"].append({
            "role": "assistant",
            "content": "Hello! I understand you're experiencing WiFi connectivity issues. Could you please describe the problem you're facing?"
        })
        return state
    
    last_message = state["messages"][-1]["content"]
    prompt = f"""Based on the user's description, determine if they are experiencing WiFi connectivity issues.
    User message: {last_message}
    Respond with either 'confirmed' or 'need_more_info'."""
    
    response = invoke_claude(prompt)
    
    if 'confirmed' in response.lower():
        state["issue_identified"] = True
        state["messages"].append({
            "role": "assistant",
            "content": "I'll help you troubleshoot your WiFi connectivity issue. Let me run some diagnostics."
        })
    else:
        state["messages"].append({
            "role": "assistant",
            "content": "Could you provide more details about your WiFi connectivity issues? Are you experiencing disconnections or slow speeds?"
        })
    
    return state

def run_diagnostics(state: AgentState) -> AgentState:
    """Run WiFi diagnostics."""
    logger.info("Running diagnostics...")
    
    if not state["issue_identified"]:
        return state
    
    prompt = """Generate a diagnostic analysis for WiFi issues covering:
    1. Network adapter status check
    2. DHCP lease verification
    3. Signal strength assessment
    """
    
    diagnostic_results = invoke_claude(prompt)
    
    state["diagnostics_complete"] = True
    state["messages"].append({
        "role": "assistant",
        "content": f"Diagnostic Results:\n{diagnostic_results}\n\nWould you like me to proceed with applying the recommended fixes?"
    })
    
    return state

def apply_resolution(state: AgentState) -> AgentState:
    """Apply resolution steps."""
    if not state["user_approval"]:
        return state
    
    resolution_steps = """
    Applying the following fixes:
    1. Flushing DNS cache
    2. Renewing DHCP lease
    3. Resetting network adapter
    """
    
    state["resolution_complete"] = True
    state["messages"].append({
        "role": "assistant",
        "content": f"{resolution_steps}\n\nPlease follow these steps to reconnect:\n1. Go to device settings\n2. Enable WiFi\n3. Select your network and connect"
    })
    
    return state

def verify_resolution(state: AgentState) -> AgentState:
    """Verify if the issue is resolved."""
    if not state["resolution_complete"]:
        return state
    
    state["messages"].append({
        "role": "assistant",
        "content": "Is your WiFi connection working properly now?"
    })
    
    return state

def create_workflow():
    """Create the workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("identify_issue", identify_issue)
    workflow.add_node("run_diagnostics", run_diagnostics)
    workflow.add_node("apply_resolution", apply_resolution)
    workflow.add_node("verify_resolution", verify_resolution)
    
    # Set entry point
    workflow.set_entry_point("identify_issue")
    
    # Add edges
    workflow.add_edge("identify_issue", "run_diagnostics")
    workflow.add_edge("run_diagnostics", "apply_resolution")
    workflow.add_edge("apply_resolution", "verify_resolution")
    workflow.add_edge("verify_resolution", END)
    
    return workflow.compile()

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.workflow = create_workflow()
        st.session_state.current_state = {
            "messages": [],
            "current_step": "identify_issue",
            "issue_identified": False,
            "diagnostics_complete": False,
            "user_approval": False,
            "resolution_complete": False
        }

def create_streamlit_ui():
    """Create Streamlit UI."""
    st.title("WiFi Troubleshooting Assistant")
    
    initialize_session_state()
    
    # Display chat history
    for message in st.session_state.current_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.current_state["messages"].append(user_message)
        
        try:
            # Process through workflow
            next_state = st.session_state.workflow.invoke(st.session_state.current_state)
            
            # Update state
            st.session_state.current_state = next_state
            
            # Handle user approval for fixes
            if "proceed" in prompt.lower() or "yes" in prompt.lower():
                st.session_state.current_state["user_approval"] = True
            
            # Add assistant response if any
            if "messages" in next_state:
                new_messages = [msg for msg in next_state["messages"] 
                              if msg not in st.session_state.current_state["messages"]]
                for msg in new_messages:
                    st.session_state.current_state["messages"].append(msg)
        
        except Exception as e:
            logger.error(f"Error: {e}")
            st.error("An error occurred. Please try again.")
        
        st.rerun()

if __name__ == "__main__":
    create_streamlit_ui()