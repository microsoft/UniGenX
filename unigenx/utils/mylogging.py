import os
from accelerate.logging import get_logger
import logging
from collections import defaultdict

class Logger:
    def __init__(self, accelerator, log_path):
        self.logger = get_logger('Main')
        self.accelerator = accelerator

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s', ""))
        self.logger.logger.addHandler(handler)
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

    def log_stats(self, stats, step, args, prefix=''):
        print("in log :", prefix)
        if args.wandb:
            if self.accelerator.is_main_process:
                self.accelerator.log({f'{prefix}{k}': v for k, v in stats.items()}, step=step)

        msg_start = f'[{prefix[:-1]}] Step {step} out of {args.total_training_steps}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.3f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)
        
    def log_message(self, msg):
        self.logger.info(msg)
        
        
class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats