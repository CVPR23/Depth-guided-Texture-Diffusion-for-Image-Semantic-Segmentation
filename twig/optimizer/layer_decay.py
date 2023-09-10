from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.logging import print_log
from nest import export
from typing import Optional

@export
@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    '''vit layer decay'''
    def __init__(self, optim_wrapper_cfg: Optional[dict]=None, paramwise_cfg: Optional[dict]=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg=None)
        self.decay_factor = paramwise_cfg.get('decay_factor', 0.75)

        super().__init__(optim_wrapper_cfg, paramwise_cfg)

    def add_params(self, params, module, prefix='' ,lr=None):
        if lr is None:
            lr = self.base_lr
        if hasattr(module, "module"):
            module_without_ddp = module.module
        else:
            module_without_ddp = module
        # # params in higher_encoder
        params_groups_vit_higher = param_groups_lrd(module_without_ddp.higher_encoder, 0.1,
            no_weight_decay_list=module_without_ddp.higher_encoder.no_weight_decay(),
            layer_decay=self.decay_factor
        )      
        for param_group in params_groups_vit_higher:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr * param_group['lr_scale'] #* 0.25
            else:
                param_group['lr'] = lr #* 0.25
        # params for space_encoder
        # params_groups_vit_space = param_groups_lrd(module_without_ddp.space_encoder, 0.1,
        #     no_weight_decay_list=module_without_ddp.space_encoder.no_weight_decay(),
        #     layer_decay=self.decay_factor
        # )      
        # for param_group in params_groups_vit_space:
        #     if 'lr_scale' in param_group:
        #         param_group['lr'] = lr * param_group['lr_scale'] #* 0.25
        #     else:
        #         param_group['lr'] = lr #* 0.25
        # params for others
        for name, param in module_without_ddp.named_parameters():
            param_group = {}
            param_group['params'] = [param]
            if name.startswith('higher_encoder'): #or name.startswith('space_encoder'):
                continue
            if not param.requires_grad:
                continue   
            else:
                param_group['lr'] = lr 
            params.append(param_group)  
        # params += params_groups_vit_space
        params += params_groups_vit_higher

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):

    param_group_names = {}
    param_groups = {}
    # num_layers = len(model.layers) 
    num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if n.startswith('norm'):
            continue
        if not p.requires_grad:
            continue
        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
        layer_id = get_layer_id_for_vit(n, num_layers)

        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)


    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
 
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    # elif name.startswith('layers'):
    #     return int(name.split('.')[1]) + 1
    # elif name.startswith('cross'):
    #     return 12
    else:
        return 12