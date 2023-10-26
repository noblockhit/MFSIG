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
* execute: `sudo systemctl restart dnsmasq`
