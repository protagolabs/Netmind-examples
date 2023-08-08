import os

env_list = ['local', 'netmind'] # local netmind
env_suffix_dict = {
    'local': 'raw',
    'netmind': 'automated'
}

compress_dict=[
        {"pytorch/language-modeling/{}" : ("trainer_customer_ddp", "torch_mlm_custom_trainer_ddp_{}.tar.gz")},

        {"pytorch/language-modeling/{}" : ("trainer_customer_no_ddp", "torch_mlm_custom_trainer_no_ddp_{}.tar.gz")},

        {"pytorch/language-modeling/{}" : ("trainer_Huggince", "torch_mlm_transformers_trainer_{}.tar.gz")},

        {"pytorch/resnet/{}": (f"trainer_customer_ddp", "torch_resnet_custom_ddp_{}.tar.gz")},

        {"pytorch/resnet/{}": (f"trainer_customer_no_ddp", "torch_resnet_custom_no_ddp_{}.tar.gz")},

        {"tensorflow/{}": ("language-modeling" , "tf-mlm-trainer-{}.tar.gz")},

        {"tensorflow/{}": ("image-classification-custom", "tf-resnet-custom-{}.tar.gz")},

        {"tensorflow/{}": ("image-classification", "tf-resnet-trainer-{}.tar.gz")}
]

copy_dir = "/Users/yang.li/Desktop/example"
root_dir = os.getcwd()

for env in env_list:
    for info in compress_dict:
        for k,v in info.items():

                assert len(v) == 2
                target_dir = k
                compress_dir = v[0]
                compress_file = v[1]
                suffix = env_suffix_dict[env]
                compress_file = compress_file.format(suffix)

                command = f"cd {target_dir.format(env)}" \
                          f"&& tar czvf {compress_file} {compress_dir}  " \
                          f"&& cp {compress_file} {copy_dir } " \
                          f"&& rm  {compress_file}" \
                          f"&& cd {root_dir}"
                ret = os.system(command)
                if ret != 0:
                    print(f'command : {command} executed failed, ret : {ret}')
                    raise

                print(f'command : {command} executed sucessfully, ret : {ret}')
