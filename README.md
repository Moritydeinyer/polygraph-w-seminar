# GSR-Based Polygraph Experimental Pipeline

## Overview
This project investigates how physical factors, specifically **applied pressure** and **moisture**, influence the measurement accuracy of a polygraph based on **Galvanic Skin Response (GSR)**. The pipeline includes sensor integration, automated data collection, and analysis to determine optimal conditions for signal detection.

---

## Research Question
How do physical factors—especially **pressure** and **moisture**—affect the measurement accuracy of a GSR-based polygraph?

---

## Objective
The goal of this experiment is to explore the dependence of GSR readings on physical influences such as **humidity** and **applied pressure**, and to identify the conditions under which the largest signal changes (∆GSR) occur. Insights from this experiment can inform the reliability of polygraph measurements under varying conditions.

---

## Experimental Setup
- **Central Unit:** Raspberry Pi Zero 2W  
- **Sensor:** GSR sensor to measure skin (or skin substitute) conductivity  
- **Skin Substitute:** Leather pad with adjustable moisture levels  
- **Pressure Variation:** Applied via weights (10 g, 50 g, 100 g, 200 g)  
- **Moisture Simulation:** 0.9% NaCl solution to mimic sweat  

---

## Measurement Principle & Hypothesis
The GSR sensor measures changes in electrical resistance (or conductivity) of the skin. Stress or emotional arousal increases sweat production, raising skin conductivity.  

**Hypothesis:**  
- Increased moisture (simulated sweat) and optimal pressure lead to larger signal changes (∆GSR).  
- Excessive pressure reduces contact resistance but may distort sensor sensitivity, decreasing accuracy.

---

## Procedure
1. **Prepare 0.9% NaCl solution:**  
   - Dissolve 0.9 g NaCl in 100 ml distilled water.  
   - Stir well and store in a clean container.  
2. **Moisturize the leather pad** at three levels: 10%, 20%, 30%.  
3. **Place the GSR sensor** on the leather pad and apply weights to vary pressure.  
4. **Measure GSR readings** for each combination of moisture and pressure for 10 seconds per trial.  
5. **Baseline measurement:** 10 g pressure and 10% moisture.  
6. **Calculate ∆GSR** as deviation from baseline for each condition.  

---

## Data Handling
- **Storage:** Measurements automatically saved in CSV files.  
- **Analysis:** Python-based tool calculates:  
  - Mean and variance per condition  
  - ∆GSR trends across pressure and moisture  
  - Visualization using 2D plots  
  - Optimal pressure for maximum signal  
  - Error analysis (measurement noise, reproducibility, statistical uncertainty)  

---

## Expected Results
- GSR readings increase with higher moisture due to increased conductivity.  
- Optimal pressure exists where the sensor maintains stable contact without excessive force.  
- Excessive pressure may reduce accuracy by mechanically affecting the sensor.  
- Results will demonstrate that polygraph measurements are influenced not only by psychological factors but also by physical conditions.

---

## Conclusion
This experiment illustrates that polygraphs are highly sensitive to **physical influences**. Even small variations in pressure or moisture can significantly affect readings, highlighting the importance of controlled experimental conditions.

---

## Requirements
- Raspberry Pi Zero 2W  
- GSR Sensor (compatible with Pi)  
- Python 3.10+  
- Libraries: `pandas`, `matplotlib`, `numpy`, `time`  

---

## Installation
```bash
pip install -r requirements.txt
