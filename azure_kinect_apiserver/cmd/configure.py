from azure_kinect_apiserver.common import KinectSystemCfg


def main(args):
    KinectSystemCfg.configure_from_keyboard()


def entry_point(argv):
    main(None)


if __name__ == '__main__':
    import sys

    entry_point(argv=sys.argv)
