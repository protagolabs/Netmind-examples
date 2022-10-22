#  this is only for netmind platform dev #
#  set your node info here #
tf_config = {
        'cluster': {
            'worker' : ['192.168.1.16:30000', '192.168.1.16:30001'],
        },
        'task': {'type': 'worker'}
}