import os

env = 'netmind' # local netmind
env_suffix_dict = {
    'local': 'raw',
    'netmind': 'automated'
}
suffix = env_suffix_dict[env] # raw
compress_dict=[
        #{f"hivemind/language-modeling/{env}" : ("trainer_customer" , "hivemind_mlm_custom_raw.tar.gz")},

        #{f"hivemind/language-modeling/{env}" : ("trainer_Huggince" ,"hivemind_mlm_trainer_raw.tar.gz")},

        #{f"hivemind/resnet" : (f"{env}", "hivemind-resnet-raw.tar.gz")},

        {f"pytorch/language-modeling/{env}" : ("trainer_customer", f"torch_mlm_custom_trainer_{suffix}.tar.gz")},

        {f"pytorch/language-modeling/{env}" : ("trainer_Huggince", f"torch_mlm_transformers_trainer_{suffix}.tar.gz")},

        {"pytorch/resnet": (f"{env}", f"torch_resnet_custom_{suffix}.tar.gz")},

        {f"tensorflow/{env}": ("language-modeling" , f"tf-mlm-trainer-{suffix}.tar.gz")},

        {f"tensorflow/{env}": ("image-classification-custom", f"tf-resnet-custom-{suffix}.tar.gz")},

        {f"tensorflow/{env}": ("image-classification", f"tf-resnet-trainer-{suffix}.tar.gz")}
]

copy_dir = "/Users/yang.li/Desktop/example"
root_dir = os.getcwd()

for info in compress_dict:
    for k,v in info.items():
        assert len(v) == 2
        target_dir = k
        compress_dir = v[0]
        compress_file = v[1]
        command = f"cd {target_dir}" \
                  f"&& tar czvf {compress_file} {compress_dir}  " \
                  f"&& cp {compress_file} {copy_dir } " \
                  f"&& rm  {compress_file}" \
                  f"&& cd {root_dir}"
        ret = os.system(command)
        if ret != 0:
            print(f'command : {command} executed failed, ret : {ret}')
            continue
        print(f'command : {command} executed sucessfully, ret : {ret}')