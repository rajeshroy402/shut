# Indoor_Shutter_1

import jetson.inference
import jetson.utils
import argparse
import sys
import datetime as dt

from utils import query_push_log, query_all_data, get_mydb_cursor, commit_and_close, SHUTTER_OPEN, \
    SHUTTER_CLOSE
from datetime import datetime

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=jetson.inference.detectNet.Usage() +
                                        jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video output object
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv + is_headless)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

mydb, cursor = get_mydb_cursor()

n_transition_frames = 0
camera_id = '1_1'
query_last_shutter_id = 'SELECT id FROM stats_shutter ORDER BY id DESC LIMIT 1'


# process frames until the user exits
def push_data_to_log_and_shutter(cursor, date_now, time_now, event):
    query = 'SELECT * FROM stats_shutter WHERE date = %s'
    params = (date_now,)
    data = query_all_data(cursor, query, params)
    if not data:
        query = 'INSERT INTO stats_shutter (date, shutter_camera_id, shutter_open_time, shutter_close_time) ' \
                'VALUES (%s, %s, %s, %s);'
        if event == SHUTTER_OPEN:
            params = (date_now, camera_id, time_now, None)
        else:
            params = (date_now, camera_id, None, time_now)
    else:
        if event == SHUTTER_CLOSE:
            query = 'UPDATE stats_shutter SET shutter_close_time = %s WHERE date = %s and shutter_camera_id = %s;'
            params = (time_now, date_now, camera_id)
        else:
            query = 'SELECT shutter_open_time FROM stats_shutter WHERE date = %s'
            params = (date_now,)
            data = query_all_data(cursor, query, params)[0][0]
            if not data:
                query = 'UPDATE stats_shutter SET shutter_open_time = %s WHERE date = %s and shutter_camera_id = %s;'
                params = (time_now, date_now, camera_id)
            else:
                return
    query_all_data(cursor, query, params)
    mydb.commit()
    print('push data to shutter', '#############')
    last_id = query_all_data(cursor, query_last_shutter_id)[0][0]
    params = (date_now, time_now, event, 'action', camera_id, last_id)
    _ = query_all_data(cursor, query_push_log, params)
    mydb.commit()
    print('push to log', '############')


init_state = False

while True:
    # capture the next image
    now = dt.datetime.now()
    if now.hour == 23 and now.minute == 58:
        break

    img = input.Capture()

    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=opt.overlay)

    if not init_state:
        if detections:
            detected_class = detections[0].ClassID
            print('detected_class: ', detected_class)
            curr_shutter_open = detected_class == 1
            shutter_open = detected_class == 1
            today = datetime.now()
            event = SHUTTER_OPEN if curr_shutter_open else SHUTTER_CLOSE
            date_now, time_now = datetime.now().date(), datetime.now().time()
            push_data_to_log_and_shutter(cursor, date_now, time_now, event)
            init_state = True
        else:
            continue

    if detections:
        detected_class = detections[0].ClassID
        print('detected_class: ', detected_class)
        curr_shutter_open = detected_class == 1

    if shutter_open:
        if not curr_shutter_open:
            n_transition_frames += 1
        else:
            n_transition_frames = 0
    else:
        if curr_shutter_open:
            n_transition_frames += 1
        else:
            n_transition_frames = 0
    print('shutter_open', shutter_open)
    print('curr_shutter_open', curr_shutter_open)
    print('n_transition_frames:', n_transition_frames)

    if n_transition_frames >= 20:
        print('change state', '##########')
        today = datetime.now()
        if shutter_open:
            shutter_open = False
            event = SHUTTER_CLOSE
        else:
            shutter_open = True
            event = SHUTTER_OPEN

        date_now, time_now = today.date(), today.time()
        push_data_to_log_and_shutter(cursor, date_now, time_now, event)
        mydb.commit()

        n_transition_frames = 0

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    # for detection in detections:
    #     print(detection)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

commit_and_close(mydb, cursor)
