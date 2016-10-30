import itertools
import serial
import json
import os, io, sys
from datetime import datetime


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

if not sys.argv[1:]:
    print "Must provide vec_size as an argument"
    print "e.g. python serial_save.py 10"
    sys.exit()
else:
    vec_size = int(sys.argv[1])

folder_path = "/home/root/measure/"
log_filename = folder_path + "index.log"

# read index log
if os.path.exists(log_filename):
    with open(log_filename, 'r') as myfile:
        index_num = int(myfile.read()) + 1
else:
    index_num = 1

while 1:

    # open Arduino serial
    ser = serial.Serial('/dev/ttyACM0', 115200)

    # fill time/active_power vector pair
    time_vector = [None] * vec_size
    active_power = [None] * vec_size
    for i in xrange(0, vec_size):

        # read serial and get a list of strings
        serial_line = ser.readline()
        split_serial = serial_line.split(" ")

        # get current time
        time_vector[i] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # handle failures and get active power values
        if ("active_power" in split_serial[0]) and is_float(split_serial[1]):
            active_power[i] = float(split_serial[1])
        else:
            active_power[i] = 0.0

    # close serial
    ser.close()

    # create JSON
    json_dict = dict(itertools.izip(time_vector, active_power))

    # save JSON to file
    filename = folder_path + "mains" + str(index_num) + ".json"
    with io.open(filename, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(json_dict, sort_keys=True, ensure_ascii=False)))

    # save index_num
    with io.open(log_filename, 'w', encoding='utf-8') as f:
        f.write(unicode(index_num))

    # increment index
    index_num += 1