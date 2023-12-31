---
runme:
  id: 01HGGX3SX1RTZHJR92NK3CXMQZ
  version: v2.0
---

# MFSIG

Microscope focus stacking image grabber.

Installation steps:

* download using:

   - `git clone https://github.com/noblockhit/MFSIG`
   - (for experimental) do:
      + `cd MFSIG`
      + `git checkout temp_progress`

* install raspap
* change raspap website port to 5000
* execute: `echo "address=/#/1.1.1.1" | sudo tee /etc/dnsmasq.d/010_captive_portal.conf`
* (_optional_) for launching on startup edit/create a file at `sudo nano /etc/xdg/lxsession/LXDE-pi/autostart` and add `@lxterminal -e /home/<username>/<path>/<to>/<mfsig>/launch.sh`
* execute: `sudo systemctl restart dnsmasq`
