# AIFirewall

Target Anomaly Types

Our model will be trained to detect four types of network anomalies:

DoS and DDoS attacks: abnormal spikes in traffic volume from one or multiple sources 
  - using the file: Wednesday-workingHours.pcap_ISCX.csv
    
Port scanning and reconnaissance: sequential connection attempts across a range of ports
  - using the file: Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    
Brute force login attempts: high frequency failed authentication packets
  - using the file: Tuesday-WorkingHours.pcap_ISCX.csv
    
Botnet command and control beaconing: periodic low volume outbound connections to suspicious IPs
  - using the file: Friday-WorkingHours-Morning.pcap_ISCX.csv
