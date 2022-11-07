import os, sys
import yaml
import _sampler, _epsilon_module

class Json_configurer(object):
    def __init__(self, config_file_path):
        assert os.path.exists(config_file_path)

        with open(config_file_path, 'r') as stream:
            config_dict = yaml.safe_load(stream)
            stream.close()

        self.config_dict = config_dict
        self.store_attr()

    def store_attr(self):
        for k, v in self.config_dict.items():
            self.__setattr__(k, v)
        

    @property
    def dataset_kwargs(self):
        # None of the args should be `None`
        attr_list = [
            "condition_key", "max_multiplexing", "use_batch_index",
            "exp_batch_key", "delimiter", "layers", "split_key",
        ]
        return  {a:self.__getattribute__(a) for a in attr_list if self.__getattribute__(a) is not None}

    @property
    def epsilon_kwargs(self):
        # base params
        attr_list = [
            "var_dim", "time_emb_dim", "n_base_perturbs", "condition_emb_dim", 
            "use_batch_index", "activation"]
        
        # specific params
        if self.epsilon_class == "Epsilon_Linear":
            attr_list.append("hidden_size")
        elif self.epsilon_class == "Epsilon_LinearAttn":
            attr_list += ['hidden_size', 'qk_dimension', 'n_heads']
        else:
            raise ValueError("non seen `Epsilon` Module")
        
        return {a:self.__getattribute__(a) for a in attr_list if self.__getattribute__(a) is not None}

    @property
    def sampler_kwargs(self):
        attr_list = ["scheduler", "loss_type", "timesteps"]

        # scheduler kw args
        possible_scheduler_kwargs = ['beta_start' 'beta_end', 's']
        for attr in possible_scheduler_kwargs:
            if (attr in self.config_dict.keys()):
                if self.config_dict[attr] is not None:
                    attr_list.append(attr)

        return {a:self.__getattribute__(a) for a in attr_list if self.__getattribute__(a) is not None}
