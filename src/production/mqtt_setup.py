"""
MQTT Broker Setup for Hummingbot Integration
"""
import subprocess
import sys
import os
from pathlib import Path

def install_mosquitto_windows():
    """Install Mosquitto MQTT broker on Windows"""
    print("Setting up Mosquitto MQTT broker...")
    
    # Check if chocolatey is available
    try:
        subprocess.run(["choco", "--version"], check=True, capture_output=True)
        print("Installing Mosquitto via Chocolatey...")
        subprocess.run(["choco", "install", "mosquitto", "-y"], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Chocolatey not found. Please install Mosquitto manually:")
        print("1. Download from: https://mosquitto.org/download/")
        print("2. Or use: winget install EclipseFoundation.Mosquitto")
        return False

def create_mosquitto_config():
    """Create basic Mosquitto configuration"""
    config_content = """
# Basic Mosquitto configuration for Hummingbot integration
port 1883
listener 1883 localhost

# Allow anonymous connections (for development only)
allow_anonymous true

# Logging
log_dest file mosquitto.log
log_type error
log_type warning
log_type notice
log_type information

# Persistence
persistence true
persistence_location ./mosquitto_data/

# Security (disable for local development)
# password_file pwfile
# acl_file aclfile
"""
    
    config_path = Path("mosquitto.conf")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    # Create data directory
    os.makedirs("mosquitto_data", exist_ok=True)
    
    print(f"Created Mosquitto config at {config_path}")
    return config_path

def install_python_dependencies():
    """Install required Python packages"""
    packages = [
        "paho-mqtt",
        "schedule"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

def start_mosquitto_service(config_path):
    """Start Mosquitto broker"""
    print("Starting Mosquitto broker...")
    print("Run this command in a separate terminal:")
    print(f"mosquitto -c {config_path}")
    print("\nOr on Windows with default installation:")
    print("net start mosquitto")

def test_mqtt_connection():
    """Test MQTT connection"""
    test_script = """
import paho.mqtt.client as mqtt
import time
import json

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe("hbot/predictions/+/ML_SIGNALS")

def on_message(client, userdata, msg):
    print(f"Received message on {msg.topic}: {msg.payload.decode()}")

# Test client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect("localhost", 1883, 60)
    client.loop_start()
    
    # Publish test message
    test_signal = {
        "timestamp": "2025-01-01T12:00:00",
        "probabilities": [0.2, 0.3, 0.5],
        "target_pct": 0.02
    }
    
    client.publish("hbot/predictions/btcusdt/ML_SIGNALS", json.dumps(test_signal))
    print("Published test signal")
    
    time.sleep(2)
    client.loop_stop()
    client.disconnect()
    print("MQTT test completed successfully!")
    
except Exception as e:
    print(f"MQTT test failed: {e}")
"""
    
    with open("test_mqtt.py", "w") as f:
        f.write(test_script)
    
    print("Created test_mqtt.py - run this to test your MQTT setup")

def main():
    """Main setup function"""
    print("=== Hummingbot MQTT Integration Setup ===\n")
    
    # Install Python dependencies
    print("1. Installing Python dependencies...")
    install_python_dependencies()
    
    # Setup Mosquitto
    print("\n2. Setting up Mosquitto MQTT broker...")
    if sys.platform == "win32":
        install_mosquitto_windows()
    else:
        print("For Linux/Mac, install Mosquitto using your package manager:")
        print("Ubuntu/Debian: sudo apt-get install mosquitto mosquitto-clients")
        print("macOS: brew install mosquitto")
    
    # Create config
    print("\n3. Creating Mosquitto configuration...")
    config_path = create_mosquitto_config()
    
    # Create test script
    print("\n4. Creating test script...")
    test_mqtt_connection()
    
    print("\n=== Setup Complete ===")
    print("\nNext steps:")
    print("1. Start Mosquitto broker:")
    print(f"   mosquitto -c {config_path}")
    print("2. Test the connection:")
    print("   python test_mqtt.py")
    print("3. Run your prediction service:")
    print("   python realtime_predictor.py")
    print("4. Configure Hummingbot to use the AI livestream controller")

if __name__ == "__main__":
    main()