#!/bin/bash
sudo iptables -t nat -F # delete all previously set rules in nat table
sudo iptables -t nat -A PREROUTING -s 10.3.141.0/24 -p tcp --dport 80 -j DNAT --to-destination 10.3.141.1:80
sudo iptables -t nat -A POSTROUTING -j MASQUERADE

BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "${BASEDIR}"

cd $BASEDIR

echo "Executing in ${PWD}"

sudo python main.py
