import os, sys 
from torch.utils.tensorboard import SummaryWriter 
import time 
import subprocess 
import path_analytic_tool 

def directory_setting(game_name, monkey_name, task_info, run_num) :
    run_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_dir = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/tensorboard_Data/')
    data_dir = os.path.join(os.path.join(os.path.expanduser('~'), f'Desktop/model_Data/{game_name}/{monkey_name}/{str(run_time)}/'))
    saver_dir = os.path.join(os.path.join(os.path.expanduser('~'), f'Desktop/model_Data/{game_name}/{monkey_name}/'))
    
    if not(os.path.exists(data_dir)):
        os.makedirs(data_dir)
    port = 6145 

    subprocess.Popen(f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)
    subprocess.Popen(f"tensorboard --logdir={log_dir} --port={port} --reload_multifile=true", shell=True)
    log_dir = log_dir + f"{game_name}/{monkey_name}/{str(run_time)}"
    log_text_dir = log_dir + "log.txt"
    writer = SummaryWriter(log_dir = log_dir)

    if not(os.path.exists(log_text_dir)):
        with open(log_text_dir, 'w') as log_file:
            log_file.write("Training log of monkey maze env \n")
    
    with open(log_text_dir, 'a') as log_file:
        log_file.write(task_info + '\n') 

    data_saver = path_analytic_tool.DataSaver(data_dir = saver_dir, model_name = game_name, run_num = run_num) 

    return data_dir, data_saver , writer 

