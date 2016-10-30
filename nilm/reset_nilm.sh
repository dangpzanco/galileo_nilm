#!/bin/bash
# Reset Measurement and NILM services

# Stop services
systemctl stop nilm.service
systemctl stop mains_measure.service

# Remove Measurement files
rm /home/root/measure/mains*
rm /home/root/measure/index.log

# Remove NILM files
rm /home/root/nilm/kettle/kettle*
rm /home/root/nilm/buffers.npz
rm /home/root/nilm/index.log

# Restart services
systemctl start mains_measure.service
systemctl start nilm.service

