import sys

import azure_kinect_apiserver.apiserver as apiserver
import azure_kinect_apiserver.cmd as cmd

args = sys.argv[1:]
if len(args) == 0:
    exit(print("No arguments provided"))
if args[0] == "configure":
    exit(cmd.configure(args[1:]))
elif args[0] == "apiserver":
    exit(apiserver.serve(args[1:]))
elif args[0] == "calibration":
    exit(cmd.calibration(args[1:]))
elif args[0] == "decode":
    exit(cmd.decode(args[1:]))
elif args[0] == "multical":
    exit(cmd.multical(args[1:]))
elif args[0] == "analyze":
    exit(cmd.analyze(args[1:]))
else:
    print("Unknown command: {}".format(args[0]))
