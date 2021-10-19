from torch.utils.tensorboard import SummaryWriter
import pytz
import datetime
import json
import os


class Logger(object):
	def __init__(self, args, dirs=None, run_tag=''):
		self.args = args

		self.run_tag = run_tag
		if dirs is None:
			self.dirs = self._make_dirs()

		self.writer = SummaryWriter(self.dirs['base'])
		self._store_args()

	def _make_dirs(self):
    if not os.path.exists('results'):
        os.mkdir('results')

    timezone = pytz.timezone('Europe/Stockholm')
    date_now = datetime.datetime.now(tz=timezone).strftime("%d-%b-%Y_%H:%M:%S")
    run_name = date_now + '|' + self.args['environment'] + self.run_tag

    dirs = {}
    dirs['base'] = os.path.join('results', run_name)
    dirs['checkpoints'] = os.path.join(dirs['base'], 'checkpoints')
    dirs['games'] = os.path.join(dirs['base'], 'games')

    os.mkdir(dirs['base'])
    os.mkdir(dirs['checkpoints'])
    os.mkdir(dirs['games'])
    return dirs

	def add_value(self, value, tag, it):
		self.writer.add_scalar(tag, value, it)

	def _store_args(self, dir=None):
		if dir is None:
			dir = self.dirs['base']
		json.dump(self.args, open(os.path.join(dir, 'arguments.json'), 'w'), indent=2)

	def add_checkpoint(self, net, args, dir=None):
		if dir is None:
			dir = self.dirs['checkpoints']





