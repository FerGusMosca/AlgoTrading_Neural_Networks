from logging.handlers import TimedRotatingFileHandler
import configparser
import logging
import os
from framework.common.logger.message_type import MessageType

class Logger:

    def __init__(self):
        self.logger = logging.getLogger("ml_research")
        self.config = configparser.ConfigParser()
        self.config.read("configs/logger.ini")
        self.level = int(self.config['DEFAULT']['level'])
        self.log_dir = self.config['DEFAULT']['log_dir']
        self.when_to_rotate = self.config['DEFAULT']['when_to_rotate']
        self.backup_count = int(self.config['DEFAULT']['backup_count'])
        self.log_file_name = self.config['DEFAULT']['log_file_name']

    def use_timed_rotating_file_handler(self):
        """

        """
        if self.level is None:
            self.level = logging.INFO

        log_path = os.path.join(self.log_dir, self.log_file_name)

        main_formatter = logging.Formatter(
            fmt='%(asctime)s [%(module)s %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        console_handler = logging.StreamHandler()
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when=self.when_to_rotate, backupCount=self.backup_count)

        for handler in [console_handler, file_handler]:
            handler.setFormatter(main_formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(self.level)

    def print(self, msg, msg_type):
        """

        Args:
            msg ():
            msg_type ():
        """
        if msg_type == MessageType.CRITICAL:
            self.logger.critical(msg)
        if msg_type == MessageType.ERROR:
            self.logger.error(msg)
        if msg_type == MessageType.WARNING:
            self.logger.warning(msg)
        if msg_type == MessageType.INFO:
            self.logger.info(msg)
        if msg_type == MessageType.DEBUG:
            self.logger.debug(msg)

    def do_log(self, msg, msg_type):
        print(msg)
        self.print(msg,msg_type)
