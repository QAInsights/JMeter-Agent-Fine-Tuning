#!/bin/bash

# Deploy script for JMeter Model API
# This script helps to start, stop, and check the status of the Flask API server

# Configuration
VENV_DIR="jmeter_venv"
API_SCRIPT="jmeter_model_api.py"
PID_FILE="api_server.pid"
LOG_FILE="api_server.log"
PORT=8080

# Function to activate virtual environment
activate_venv() {
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
}

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null; then
            return 0  # Running
        else
            rm "$PID_FILE"  # PID file exists but process is not running
            return 1
        fi
    else
        return 1  # Not running
    fi
}

# Function to start the server
start_server() {
    if is_running; then
        echo "Server is already running with PID $(cat $PID_FILE)"
        return
    fi
    
    echo "Starting JMeter Model Flask API server..."
    activate_venv
    
    # Start the server in the background
    python "$API_SCRIPT" > "$LOG_FILE" 2>&1 &
    pid=$!
    echo $pid > "$PID_FILE"
    echo "Server started with PID $pid"
    echo "Log file: $LOG_FILE"
    echo "API available at: http://localhost:$PORT"
    echo "\nAPI Endpoints:\n- POST /predict - Predict JMeter element from query\n- GET /health - Health check endpoint"
}

# Function to stop the server
stop_server() {
    if is_running; then
        pid=$(cat "$PID_FILE")
        echo "Stopping server with PID $pid..."
        kill "$pid"
        rm "$PID_FILE"
        echo "Server stopped"
    else
        echo "Server is not running"
    fi
}

# Function to check server status
server_status() {
    if is_running; then
        pid=$(cat "$PID_FILE")
        echo "Server is running with PID $pid"
        echo "API available at: http://localhost:$PORT"
        echo "\nAPI Endpoints:\n- POST /predict - Predict JMeter element from query\n- GET /health - Health check endpoint"
    else
        echo "Server is not running"
    fi
}

# Function to test the API
test_api() {
    if ! is_running; then
        echo "Server is not running. Start it first with './deploy_api.sh start'"
        return
    fi
    
    echo "Testing Flask API with sample queries..."
    activate_venv
    python test_api.py
}

# Main script logic
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        server_status
        ;;
    test)
        test_api
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test}"
        exit 1
        ;;
esac

exit 0
