#include <WiFi.h>
#include <PubSubClient.h>

// =========================================================================
// 1. FILL IN YOUR CREDENTIALS HERE
// =========================================================================
const char* WIFI_SSID     = "404cluenotfound";       
const char* WIFI_PASS     = "SyntaxLala";   
const char* AIO_SERVER    = "io.adafruit.com";
const int   AIO_PORT      = 1883;
const char* AIO_USERNAME  = "Pibie";    
const char* AIO_FEED_SUB  = "smartgrid";

// =========================================================================
// 2. HARDWARE PINS & CALIBRATION FACTORS
// =========================================================================
const int transVoltagePin = 35;
const int transCurrentPin = 34;
const int consVoltagePin  = 33;
const int consCurrentPin  = 32;

// Transformer Sensors
const float TRANS_V_CAL   = 0.8996; 
const float TRANS_C_INTER = 9047.72;  
const float TRANS_C_SLOPE = -9.218;   

// Consumer Sensors 
const float CONS_V_CAL    = 1.1518; 
const float CONS_C_INTER  = 6067.29;  
const float CONS_C_SLOPE  = -4.770;   

unsigned long lastTime = 0;
unsigned long lastMQTTSend = 0;
double transEnergy_Wh = 0.0;
double consEnergy_Wh  = 0.0;
double energyLoss_Wh  = 0.0; // Acts as a cumulative theft tracker

// --- MQTT Averaging Accumulators ---
float sumTransVoltage = 0.0;
float sumTransCurrent = 0.0;
float sumTransPower   = 0.0;
float sumConsVoltage  = 0.0;
float sumConsCurrent  = 0.0;
float sumConsPower    = 0.0;
float sumPowerLoss    = 0.0;
float sumRawLoss      = 0.0;
int sampleCount       = 0;

WiFiClient espClient;
PubSubClient mqttClient(espClient);
char mqttTopic[128];

float getAveragedADC(int pin) {
  long sum = 0;
  for (int i = 0; i < 20; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  return sum / 20.0;
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("\nConnecting to WiFi '");
  Serial.print(WIFI_SSID);
  Serial.print("'");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 25) {
    delay(500);
    Serial.print(".");
    retry++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected! IP: " + WiFi.localIP().toString());
  }
}

void connectMQTT() {
  if (mqttClient.connected()) return;
  connectWiFi();
  Serial.println("[MQTT] Connecting to io.adafruit.com...");
  String clientId = "ESP32_GridNest_" + String(random(0xffff), HEX);
  if (mqttClient.connect(clientId.c_str(), AIO_USERNAME, AIO_KEY)) {
    Serial.println("[MQTT] Connected to Adafruit IO successfully!");
  } else {
    Serial.printf("[MQTT] Connect failed, rc=%d. Retrying...\n", mqttClient.state());
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(transVoltagePin, INPUT);
  pinMode(transCurrentPin, INPUT);
  pinMode(consVoltagePin, INPUT);
  pinMode(consCurrentPin, INPUT);
  analogReadResolution(12);

  lastTime = millis();
  lastMQTTSend = millis();
  snprintf(mqttTopic, sizeof(mqttTopic), "%s/feeds/%s", AIO_USERNAME, AIO_FEED_SUB);

  mqttClient.setServer(AIO_SERVER, AIO_PORT);
  mqttClient.setBufferSize(512);

  connectWiFi();
  connectMQTT();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  // Read ADC
  float transV_Raw = getAveragedADC(transVoltagePin);
  float transC_Raw = getAveragedADC(transCurrentPin);
  float consV_Raw  = getAveragedADC(consVoltagePin);
  float consC_Raw  = getAveragedADC(consCurrentPin);

  // Calibrate V & I
  float transVoltage = (transV_Raw / 4095.0) * 3.3 * 5.0 * TRANS_V_CAL;
  if (transVoltage < 0.1) transVoltage = 0.0;
  
  float transCurrent = TRANS_C_INTER + (TRANS_C_SLOPE * transC_Raw);
  if (transCurrent < 100 && transCurrent > -100) transCurrent = 0.0;
  transCurrent = abs(transCurrent);

  float consVoltage = (consV_Raw / 4095.0) * 3.3 * 5.0 * CONS_V_CAL;
  if (consVoltage < 0.1) consVoltage = 0.0;
  
  float consCurrent = CONS_C_INTER + (CONS_C_SLOPE * consC_Raw);
  if (consCurrent < 100 && consCurrent > -100) consCurrent = 0.0;
  consCurrent = abs(consCurrent);

  // Time & Power
  unsigned long currentTime = millis();
  float deltaTime_hours = (currentTime - lastTime) / 3600000.0;
  lastTime = currentTime;

  float transPower_W = transVoltage * (transCurrent / 1000.0);
  float consPower_W  = consVoltage * (consCurrent / 1000.0);
  
  // Flipped subtraction to match your physical desk setup:
  // Now it tracks when the Consumer sensor reads higher than the Transformer
  float rawLoss_W = consPower_W - transPower_W;
  float powerLoss_W = 0.0;

  // THRESHOLD FILTER: Ignore minor calibration differences. 
  // If the Consumer spikes more than 0.20W above the Transformer, flag as theft.
  if (rawLoss_W > 0.20) {
    powerLoss_W = rawLoss_W;
  }

  transEnergy_Wh += (transPower_W * deltaTime_hours);
  consEnergy_Wh  += (consPower_W * deltaTime_hours);
  
  // Energy loss will now strictly remain at 0.0000 until theft exceeds the threshold
  energyLoss_Wh  += (powerLoss_W * deltaTime_hours); 

  // Accumulate readings for MQTT payload
  sumTransVoltage += transVoltage;
  sumTransCurrent += transCurrent;
  sumTransPower   += transPower_W;
  sumConsVoltage  += consVoltage;
  sumConsCurrent  += consCurrent;
  sumConsPower    += consPower_W;
  sumPowerLoss    += powerLoss_W; 
  sumRawLoss      += rawLoss_W; 
  sampleCount++;

  // Publish to Adafruit IO every 2.5 seconds using Averaged Data
  if (currentTime - lastMQTTSend >= 2500) {
    lastMQTTSend = currentTime;
    
    if (sampleCount > 0) {
      float avgTransVoltage = sumTransVoltage / sampleCount;
      float avgTransCurrent = sumTransCurrent / sampleCount;
      float avgTransPower   = sumTransPower / sampleCount;
      float avgConsVoltage  = sumConsVoltage / sampleCount;
      float avgConsCurrent  = sumConsCurrent / sampleCount;
      float avgConsPower    = sumConsPower / sampleCount;
      
      float avgPowerLoss    = sumPowerLoss / sampleCount; 
      float avgRawLoss      = sumRawLoss / sampleCount;

      char payload[384];
      snprintf(payload, sizeof(payload),
        "{\"transVoltage\":%.2f,\"transCurrent\":%.2f,\"transPower\":%.2f,\"transEnergy\":%.4f,"
        "\"consVoltage\":%.2f,\"consCurrent\":%.2f,\"consPower\":%.2f,\"consEnergy\":%.4f,"
        "\"powerLoss\":%.2f,\"energyLoss\":%.4f}",
        avgTransVoltage, avgTransCurrent, avgTransPower, transEnergy_Wh,
        avgConsVoltage, avgConsCurrent, avgConsPower, consEnergy_Wh,
        avgPowerLoss, energyLoss_Wh
      );

      if (mqttClient.connected()) {
        bool sent = mqttClient.publish(mqttTopic, payload);
        if (sent) {
            Serial.println();
            Serial.printf("[Adafruit IO] Packet Sent! (Avg of %d samples)\n", sampleCount);
            Serial.println("----------------------");
            Serial.printf("Transformer Voltage : %.2f V , Transformer Current : %.2f mA , Transformer Power : %.2f W , Transformer Energy : %.4f Wh\n", 
                          avgTransVoltage, avgTransCurrent, avgTransPower, transEnergy_Wh);
            Serial.printf("Consumer Voltage : %.2f V , Consumer Current : %.2f mA , Consumer Power : %.2f W , Consumer Energy : %.4f Wh\n", 
                          avgConsVoltage, avgConsCurrent, avgConsPower, consEnergy_Wh);
            
            Serial.printf("[diag] Signed Avg Loss: %.2f W\n", avgRawLoss);
            Serial.printf("Reported Dashboard Loss : %.2f W , Energy Loss : %.4f Wh\n", 
                          avgPowerLoss, energyLoss_Wh);
            Serial.println("----------------------\n");
        }
      }

      // Reset the accumulators for the next 2.5 second window
      sumTransVoltage = 0.0;
      sumTransCurrent = 0.0;
      sumTransPower   = 0.0;
      sumConsVoltage  = 0.0;
      sumConsCurrent  = 0.0;
      sumConsPower    = 0.0;
      sumPowerLoss    = 0.0;
      sumRawLoss      = 0.0;
      sampleCount     = 0;
    }
  }

  delay(200);
}
