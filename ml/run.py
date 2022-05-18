import logging
from learn import run
from util import get_parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = get_parser()
    args = parser.parse_args()

    run(args)
