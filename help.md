## Setup for a different host/user:

__host__ change the dispatcher/hosts file to contain the IP of the working node (s?) ie the one that will run the samples in the cuckoo environment based on ansible playbook

In our case we are connecting to a headless VirtualBox image. To start a VirtualBox image in this manner, run `VBoxManage startvm /path/to/vmdir --type headless`

If after the vm is started you would like to view the guest information (guest additions required) such as IP address for connecting over ssh, run `VBoxManage guestproperty enumerate /path/to/vmdir`. If guest additions are not available, you might grep for IP in the vmname.vbox XML file inside vmdir.

__user__ be sure to change ansible_user in dispatcher/hosts as well as cuckoo_user in 
