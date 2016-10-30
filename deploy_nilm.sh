#!/bin/bash

# arrumar DNS
# ethernet_984fee0554ed_cable
ETHERNET="$(connmanctl services | sed 's/^.*Wired                //')"
connmanctl config ${ETHERNET} --nameservers 8.8.8.8

# atualizar repositorios
opkg update

# instalar http file server
opkg install nodejs-npm
npm install http-server -g

# instalar pip
easy_install pip

# libs python
pip install pyserial

# arrumar o timezone
# rm /etc/localtime
# ln -s /usr/share/zoneinfo/America/Sao_Paulo /etc/localtime
pip install tzupdate
tzupdate

# copiar arquivos
mkdir /home/root/measure
mkdir /home/root/nilm

cp measure/mains_measure.py /home/root
cp nilm/run_nilm.py /home/root/nilm
cp nilm/kettle_model.npz /home/root/nilm
cp nilm/reset_nilm.sh /home/root/nilm

cp services/*.service /etc/systemd/system/

# iniciar serviços
systemd start mains_measure.service
systemd start node_http-server.service
systemd start mains_measure.service









