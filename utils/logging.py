"""colorful logging
# import the whole file
"""

import coloredlogs, logging
from termcolor import colored, cprint

logging.basicConfig()
logger = logging.getLogger()
coloredlogs.install(level='INFO', logger=logger)


# from sty import fg, bg, ef, rs


def toRed(text):
	return colored(text, 'red', attrs=['reverse'])


def toCyan(text):
	return colored(text, 'cyan', attrs=['reverse'])
