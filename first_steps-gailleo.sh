#!/bin/bash

# gravar no SD Card
# sudo dcfldd if=~/Documents/galileo/images/iot-devkit-prof-dev-image-galileo-20160606.direct of=/dev/mmcblk0 bs=8M statusinterval=4
# sudo fdisk /dev/mmcblk0
# sudo resize2fs /dev/mmcblk0p2

# arrumar DNS
# ethernet_984fee0554ed_cable
ETHERNET="$(connmanctl services | sed 's/^.*Wired                //')"
connmanctl config ${ETHERNET} --nameservers 8.8.8.8

#echo 'nameserver 8.8.8.8' >> /etc/resolv.conf
#reboot

# wifi
rfkill list
ifconfig wlp1s0 up

connmanctl enable wifi
connmanctl scan wifi
MY_WIFI="$(connmanctl services | grep -i TP-LINK_AB9110 | sed 's/^.*TP-LINK_AB9110 //')"
connmanctl connect ${MY_WIFI}
connmanctl config ${MY_WIFI} --nameservers 8.8.8.8


# arrumar o timezone
rm /etc/localtime
ln -s /usr/share/zoneinfo/America/Sao_Paulo /etc/localtime

# atualizar repositorios
opkg update
opkg install mraa

# remover Redis
opkg remove --force-removal-of-dependent-packages redis
opkg remove --force-removal-of-dependent-packages python-redis
opkg remove --force-removal-of-dependent-packages libhiredis0.13

# remover XDK Daemon
opkg remove --force-removal-of-dependent-packages xdk-daemon

# remover alsa-utils (audio) e openjdk-8-java
opkg remove --force-removal-of-dependent-packages alsa*

# instalar http file server
opkg install nodejs-npm
npm install http-server -g

# instalar pip
easy_install pip

# libs python
pip install pandas # não instala
pip install pyserial

# ver servicos inuteis
systemctl list-unit-files | grep enabled

systemctl disable bluetooth.service
systemctl disable wyliodrin-hypervisor.service
systemctl disable wyliodrin-server.service
systemctl disable xdk-daemon.service
systemctl disable lighttpd.service

daniel@daniel-ubuntu:~/apps/iot-devkit/1.7.3$ i586-poky-linux-gcc --sysroot=./sysroots/i586-poky-linux/ hello.c -o hi.bin

# opkg upgrade:

wpa-supplicant-dev: unsatisfied recommendation for libnl-genl-dev
wpa-supplicant-dev: unsatisfied recommendation for libssl-dev
wpa-supplicant-dev: unsatisfied recommendation for dbus-lib-dev
wpa-supplicant-dev: unsatisfied recommendation for libcrypto-dev

e2fsprogs-dev: unsatisfied recommendation for libcomerr-dev
e2fsprogs-dev: unsatisfied recommendation for e2fsprogs-badblocks-dev
e2fsprogs-dev: unsatisfied recommendation for libe2p-dev
e2fsprogs-dev: unsatisfied recommendation for update-alternatives-opkg-dev
e2fsprogs-dev: unsatisfied recommendation for libext2fs-dev
e2fsprogs-dev: unsatisfied recommendation for libss-dev







