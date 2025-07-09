/*************************************************** 
  This is a library for the Adafruit PT100/P1000 RTD Sensor w/MAX31865

  Designed specifically to work with the Adafruit RTD Sensor
  ----> https://www.adafruit.com/products/3328

  This sensor uses SPI to communicate, 4 pins are required to  
  interface
  Adafruit invests time and resources providing this open source code, 
  please support Adafruit and open-source hardware by purchasing 
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.  
  BSD license, all text above must be included in any redistribution

  Modified by F. Paz and L. Martin in 2021, 
  Linear fit coefficients *1.07993-3.27219 measured at 0 ºC and 100 ºC 
  using NIST published procedure.


  Modified by Diego Menéndez in 2025 to add a DPDT (model DR25E01) controlled by the same Arduino
  
 ****************************************************/

#include <Adafruit_MAX31865.h>

// Use software SPI: CS, DI, DO, CLK
Adafruit_MAX31865 max = Adafruit_MAX31865(10, 11, 12, 13);
// use hardware SPI, just pass in the CS pin
//Adafruit_MAX31865 max = Adafruit_MAX31865(10);

// The value of the Rref resistor. Use 430.0 for PT100 and 4300.0 for PT1000
#define RREF      430.0
#define RNOMINAL  100.0
float value;
float value1;

const int triggerPin = 2;  //DPDT DR25E01 trigger pin 
bool relayState = false;  // Initial state: false = OFF


void setup() {
  Serial.begin(115200);
  max.begin(MAX31865_2WIRE);  // set to 2WIRE or 4WIRE as necessary

  pinMode(triggerPin, OUTPUT);
  digitalWrite(triggerPin, HIGH);  // Hold pin HIGH by default
}


void loop() {

// Temperature lectures
value = 0;
for (int i = 0; i < 5; i++) {
  value = value + max.temperature(RNOMINAL, RREF);
  //delay(1);
  }
  value1 = (value/5)*1.07993-3.27219;
  
  //delay(0.5);

// Trigger of DPDT
  char command = Serial.read();

    if (command == 'T' || command == 't') {
      digitalWrite(triggerPin, LOW);  //send a pulse to the trigger
      delay(150);  //ms (lenght of the pulse)
      digitalWrite(triggerPin, HIGH);
      relayState =! relayState;    // change relay state  
    }

  Serial.print(value1,4);
  Serial.print(",");
  Serial.println(relayState ? "ON" : "OFF"); 
}