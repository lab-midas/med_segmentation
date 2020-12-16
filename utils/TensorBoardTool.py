import tensorflow as tf
from tensorboard import default
from tensorboard import program
import logging
import sys
from distutils.version import LooseVersion


class TensorBoardTool:

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.run()

    def run(self):
        if LooseVersion(tf.__version__) < LooseVersion("1.12"): # v1.10
            # Remove http messages
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            # Start tensorboard server
            tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
            tb.configure(argv=['--logdir', self.dir_path])
            url = tb.launch()
            sys.stdout.write('TensorBoard at %s \n' % url)

        elif LooseVersion(tf.__version__) >= LooseVersion("1.12") and LooseVersion(tf.__version__) < LooseVersion("2.0"):  # v1.12 - v1.14
            # Remove http messages
            log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
            # Start tensorboard server
            tb = program.TensorBoard(default.get_plugins())  #, default.get_assets_zip_provider())
            tb.configure(argv=[None, '--logdir', self.dir_path])
            url = tb.launch()
            sys.stdout.write('TensorBoard at %s \n' % url)
        else:  # v2.X
            # Remove http messages
            log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
            # Start tensorboard server
            tb = program.TensorBoard(default.get_plugins())  # , default.get_assets_zip_provider())
            tb.configure(argv=[None, '--logdir', self.dir_path])
            url = tb.launch()
            sys.stdout.write('TensorBoard at %s \n' % url)

