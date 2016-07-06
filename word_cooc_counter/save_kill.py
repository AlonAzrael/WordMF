
import os

_count = 0

def incrcounter(n):
    global _count
    _count = _count + n

def savecounter(*args):
    open("counter", "w").write("%d" % _count)
    os._exit(0)

# import atexit
# atexit.register(savecounter)

from signal import *

def clean(*args):
    print "clean me"
    sys.exit(0)

for sig in (SIGABRT, SIGINT, SIGTERM):
    signal(sig, savecounter)


_count = raw_input("input count")

try:
    _count = int(open("counter").read())
except IOError:
    _count = 0


