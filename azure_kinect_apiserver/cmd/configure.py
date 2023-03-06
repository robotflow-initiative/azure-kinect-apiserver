import argparse

from azure_kinect_apiserver.common import KinectSystemCfg


def main(args):
    print(args)
    try:
        KinectSystemCfg.configure_from_keyboard()
    except KeyboardInterrupt:
        print("got KeyboardInterrupt")


def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    main(argv)


if __name__ == '__main__':
    import sys

    entry_point(argv=sys.argv)
