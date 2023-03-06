import socket
import getpass
import os

username = getpass.getuser()
hostname = socket.gethostname()

if hostname=='SL20':
    if username=='ayadi':
        dataroot = '/mnt/hdd/ayadi/data'
        logroot = '/home/ayadi/Implicit3DUnderstanding/out'
