import sys
from bodegas import Bodegas


def main (args=None):
    bodegas = Bodegas()
    bodegas.get_popularity()

if __name__ == '__main__':
    sys.exit(main())
