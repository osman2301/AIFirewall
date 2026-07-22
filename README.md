# AIFirewall

Machine Learning-Based Network Anomaly Detection — Networks and Security term project.

This project detects anomalous network traffic using two ML approaches: a Random Forest
classifier (supervised) and an Isolation Forest (unsupervised), trained and evaluated on
the CICIDS2017 dataset.

## Target Anomaly Types

Our model will be trained to detect four types of network anomalies:

- **DoS and DDoS attacks**: abnormal spikes in traffic volume from one or multiple sources
  - using the file: `Wednesday-workingHours.pcap_ISCX.csv`
- **Port scanning and reconnaissance**: sequential connection attempts across a range of ports
  - using the file: `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- **Brute force login attempts**: high frequency failed authentication packets
  - using the file: `Tuesday-WorkingHours.pcap_ISCX.csv`
- **Botnet command and control beaconing**: periodic low volume outbound connections to suspicious IPs
  - using the file: `Friday-WorkingHours-Morning.pcap_ISCX.csv`

## Setup

### 1. Environment
This project runs inside a Python virtual environment (Ubuntu VM or your own machine both work).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset
We use the CICIDS2017 dataset. Download `MachineLearningCSV.zip` from:
https://www.unb.ca/cic/datasets/ids-2017.html → Datasets → CIC-IDS-2017 → CSVs

Unzip it, then copy the 4 files listed above (under Target Anomaly Types) into `data/raw/`
in this repo:

```bash
mkdir -p data/raw
cp path/to/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv data/raw/
cp path/to/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv data/raw/
cp path/to/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv data/raw/
cp path/to/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv data/raw/
```

### 3. Prepare the data
```bash
python prepareData.py
```
This cleans the data and saves a ready-to-use train/test split to `results/prepareData_data.joblib`.

### 4. Train and evaluate models
```bash
python training.py
```
This trains Random Forest and Isolation Forest and writes results to `results/model_comparison.csv`.
