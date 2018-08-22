import os
import time

i = 0

while (i < 20):
    print i
    cmd = "raspistill -vf -o left" + str(i) + ".jpg"
    os.system(cmd)
    i += 1

