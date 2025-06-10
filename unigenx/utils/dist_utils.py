# -*- coding: utf-8 -*-
import os


def is_master_node():
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False
