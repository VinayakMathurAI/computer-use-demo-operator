#!/bin/bash

# Explicitly set dimensions
WIDTH=1920
HEIGHT=1080

# Export all required variables
export AWS_PROFILE=default
export API_PROVIDER=bedrock
export AWS_REGION=us-west-2
export WIDTH
export HEIGHT

# Create necessary directories
mkdir -p /home/ubuntu/.streamlit
mkdir -p /home/ubuntu/.anthropic
mkdir -p /home/ubuntu/.aws

# Add Streamlit config
cat > /home/ubuntu/.streamlit/config.toml << EOF
[server]
address = "0.0.0.0"
port = 8501
enableCORS = true

[browser]
serverAddress = "0.0.0.0"
EOF

echo "Starting with resolution: ${WIDTH}x${HEIGHT}"

# Build the local image with custom changes
docker build . -t computer-use-demo_v2:local

# Run the container with custom changes
docker run \
    -e API_PROVIDER=$API_PROVIDER \
    -e AWS_PROFILE=$AWS_PROFILE \
    -e AWS_REGION=$AWS_REGION \
    -e WIDTH=$WIDTH \
    -e HEIGHT=$HEIGHT \
    -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    -v /home/ubuntu/.aws:/home/computeruse/.aws \
    -v /home/ubuntu/.anthropic:/home/computeruse/.anthropic \
    -v /home/ubuntu/.streamlit:/home/computeruse/.streamlit \
    -v $(pwd)/computer_use_demo:/home/computeruse/computer_use_demo/ \
    -p 0.0.0.0:5900:5900 \
    -p 0.0.0.0:8501:8501 \
    -p 0.0.0.0:6080:6080 \
    -p 0.0.0.0:8080:8080 \
    -it computer-use-demo:local