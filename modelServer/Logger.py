import sys

# setup logging
class Logger(object):
    def __init__(self, debug, destfile, prefix=''):
        self.debug = debug
        self.terminal = sys.stdout
        self.log = open(destfile, "a")
        self.prefix = prefix

    def write(self, message):
        self.log.write(self.prefix + " " + str(message) + "\n")
        self.log.flush()
        #if self.debug:
        #    self.terminal.write(self.prefix + str(message) + "\n")
        #    self.terminal.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.flush()
        self.terminal.flush()
        pass
