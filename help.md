## Setup for a different host/user:

__host__ change the dispatcher/hosts file to contain the IP of the working node (s?) ie the one that will run the samples in the cuckoo environment based on ansible playbook

In our case we are connecting to a headless VirtualBox image. To start a VirtualBox image in this manner, run `VBoxManage startvm /path/to/vmdir --type headless`

If after the vm is started you would like to view the guest information (guest additions required) such as IP address for connecting over ssh, run `VBoxManage guestproperty enumerate /path/to/vmdir`. If guest additions are not available, you might grep for IP in the vmname.vbox XML file inside vmdir.

__user__ be sure to change ansible_user in dispatcher/hosts as well as cuckoo_user in 


## Progress for Ubuntu20.04 host

Differences from ubuntu 16
- comment out python-pip and virtualenv
- m2crypto 0.35 required to work with pyopenssl
- comment out libvirt in kvm
- group libvirtd does not exist --> `sudo addgroup libvirtd` and `sudo adduser <cuckoo_user> libvirtd`
- change kvm vmcloak branch from wip/kvm_virsh to dev/kvm
- change vmcloak branch from master to wip/vbox

VM Backend, config, ok, changed, skipped, ignored, exection time (approx, s):
- [x] virtualbox, win7x86_conf4, 42, 12, 27, 1
- [x] vmware, win7x86_conf4, 42, 15, 27, 1 
- [x] kvm, win7x86_conf4, 47, 22, 22, 1
- [x] qemu, win7x86_conf4, 34, 9, 35, 1
- [x] esx, win7x86_conf4, 34, 9, 35, 1
- [x] xenserver, win7x86_conf4, 34, 9, 35, 1
- [x] vsphere, win7x86_conf4, 34, 9, 35, 1
- [x] avd, win7x86_conf4, 34, 9, 35, 1

## Progress for Ubuntu16.04 VirtualBox host


## updates
- dispatcher/samples/sampler.py: select from a number specified number of 

## random/known issues

- failed to update apt cache unknown reason: https://githubhot.com/repo/pythops/jetson-nano-image/issues/31
	- apt update appears to modify /etc/resolv.conf nameserver value (? trying to repro)
- vmware bundle checksum did not match 3212ed00463784ca8c67b5acd2c8d1cd, was 81e3cc66e5ce815457ae94ad52093ab3