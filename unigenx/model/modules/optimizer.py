
from typing import Callable, List, Optional, Tuple
from unigenx.logging import logger

try:
    # from torch.optim import Adam
    # logger.info("using torch adam")
    from apex.optimizers import FusedAdam as Adam  # isort:skip

    logger.info("apex is installed, using FusedAdam with fp16 optimizer states")

    def AdamW(*args, **kwargs):
        return Adam(*args, **kwargs, adam_w_mode=True)

    # from torch.optim import Adam
except:
    logger.info("apex is not installed, using pytorch AdamW with fp32 optimizer states")
    from ...utils.adam import AdamW

def split_param_and_layer_name(name_list: List[str]) -> Tuple[List[str], List[int]]:
    param_list = []
    layer_name_list = []
    for name in name_list:
        if isinstance(name, str):
            param_list.append(name)
        elif isinstance(name, int):
            layer_name_list.append(name)
        else:
            raise ValueError(f"Invalid name type: {type(name)}")

    return param_list, layer_name_list

def process_param(
    net,
    freeze_list: List = [],
    unfreeze_list: List = [],
    lr: float = 1e-5,
    mfm_lora: bool = False,
    **kwargs,
):
    param_groups = [{}]
    param_groups[0]["lr"] = lr
    param_groups[0]["params"] = []
    logger.info(f"freeze list: {freeze_list})")
    logger.info(f"unfreeze list: {unfreeze_list})")
    if len(unfreeze_list) > 1:
        unfreeze_list, unfreeze_layer_name_list = split_param_and_layer_name(
            unfreeze_list
        )
        logger.info(f"unfreeze layer name list: {unfreeze_list}")
        for name, param in net.named_parameters():
            nl = int(name.split(".")[0]) if name.split(".")[0].isdigit() else -1
            if nl in unfreeze_layer_name_list:
                param_groups[0]["params"].append(param)
                if name.find("dummy") == -1:
                    logger.info(f"unfreeze layer: {name}")
            else:
                print(name)
                for unfreeze_name in unfreeze_list:
                    if name.find(unfreeze_name) != -1:
                        param_groups[0]["params"].append(param)
                        if name.find("dummy") == -1:
                            logger.info(f"unfreeze layer: {name}")

    elif len(freeze_list) > 0:
        freeze_list, freeze_layer_name_list = split_param_and_layer_name(freeze_list)
        for name, param in net.named_parameters():
            nl = int(name.split(".")[0]) if name.split(".")[0].isdigit() else -1
            if nl in freeze_layer_name_list:
                flag = True
                logger.info(f"freeze layer: {name}")
            else:
                for freeze_name in freeze_list:
                    flag = False
                    if name.find(freeze_name) != -1:
                        flag = True
                        logger.info(f"freeze {name}")
                        break
            if not flag:
                param_groups[0]["params"].append(param)

    else:
        logger.info("unfreeze all layers")
        for name, param in net.named_parameters():
            param_groups[0]["params"].append(param)

    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)

    return param_groups

def process_parm_list(param_list: str = None):
    if not param_list:
        return []

    ret = []
    if isinstance(param_list, str):
        param_list = param_list.strip()
        for name in param_list.split(","):
            name = name.strip()
            if name:
                ret.append(name)
    return ret

def myAdamW(
    net,
    impl=AdamW,
    freeze_list=None,
    unfreeze_list=None,
    mfm_lora=False,
    **kwargs,
):
    freeze_list = process_parm_list(freeze_list)
    unfreeze_list = process_parm_list(unfreeze_list)

    assert (
        len(freeze_list) == 0 or len(unfreeze_list) == 0
    ), f"freeze_list and unfreeze_list cannot be set at the same time, got {freeze_list=}, {unfreeze_list=}"

    # When using unfreeze_list, we always want to unfreeze the dummy layer
    if len(unfreeze_list) > 0 and "dummy" not in unfreeze_list:
        unfreeze_list.append("dummy")

    new_param_groups = []
    param_groups = process_param(
        net,
        freeze_list=freeze_list,
        unfreeze_list=unfreeze_list,
        mfm_lora=mfm_lora,
        **kwargs,
    )
    for param_group in param_groups:
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs), param_groups