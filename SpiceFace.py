from subprocess import Popen
import sys

#filename = ""C:\SpiceFace\spicefaceGUI.py""
filename = "spicefaceGUI.py"

x = Popen("cd C:\\SpiceFace\\", shell=True)


while True:
    print("\nStarting " + filename)
    p = Popen("python " + filename, shell=True)
    p.wait()

#https://datatofish.com/python-script-windows-scheduler/