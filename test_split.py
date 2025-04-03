from main import *

def roda_command(director, split_fraction, model, max_iterations):
    os.system(f'python3 main.py -id {director} -sf {split_fraction} -m {model} -mi {max_iterations} -df 4 -es True')

def gather_info(director, dict_infos, split_fraction, elems, video_begin):
    for file in os.listdir(director):
        info_path = os.path.join(director, file, 'output_metrics_features.json')
        info = read_info(info_path)
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram'] = {}
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['tempo_colmap'] = info['tempo_colmap']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['gpu_colmap_max_vram'] = info['gpu_colmap_max_vram']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['gpu_colmap_max_perc'] = info['gpu_colmap_max_perc']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['ram_colmap_max'] = info['ram_colmap_max']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['splatfacto-tempo_train'] = info['splatfacto']['tempo_train']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['splatfacto-gpu_train_max_vram'] = info['splatfacto']['gpu_train_max_vram']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['splatfacto-gpu_train_max_perc'] = info['splatfacto']['gpu_train_max_perc']
        dict_infos[file][video_begin][split_fraction]['tempo_gpu_ram']['splatfacto-ram_train_max'] = info['splatfacto']['ram_train_max']

        dict_infos[file][video_begin][split_fraction]['metrics'] = {}
        evaluations_path = os.path.join(director, file, 'output_splatfacto', 'evaluations')
        for evaluation_file, elem in zip(sorted(os.listdir(evaluations_path)), elems):
            evaluation = read_info(os.path.join(evaluations_path, evaluation_file))
            dict_infos[file][video_begin][split_fraction]['metrics'][elem] = evaluation['results_list']
    return dict_infos

def delete_all(director):
    for file in os.listdir(director):
        os.system(f'rm -rf {os.path.join(director, file, "output_splatfacto")}')
        os.system(f'rm -rf {os.path.join(director, file, "report")}')
        os.system(f'rm -f {os.path.join(director, file, "output_metrics_features.json")}')
        if os.path.exists(os.path.join(director, file, "info.json")):
            info = read_info(os.path.join(director, file, "info.json"))
            info['splatfacto'] = {
                    "trained": False,
                    "evaluations": False
                }
            write_info(os.path.join(director, file, "info.json"), info)

project = 'parquinho'

dict_infos = {}
dict_infos[project] = [{},{},{},{},{}]

if os.path.exists('./sala_split.json'):
    dict_infos = read_info('./sala_split.json')

director = "/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Videos_parquinho_tudo"

max_iterations = 50000
elems = [*range(10000, max_iterations, 10000)]
elems.append(max_iterations - 1)

for video_begin in range(5):
    for split_fraction in [0.2,0.4,0.6,0.8,1.0]:
    # for split_fraction in [0.2]:
        k = round(0.04 * video_begin + split_fraction - 0.2, 2)
        dict_infos[project][video_begin][split_fraction] = {}
        delete_all(director)
        roda_command(director, k, 'splatfacto', max_iterations)
        dict_infos = gather_info(director, dict_infos, split_fraction, elems, video_begin)
        # delete_all(director)
        write_info('./sala_split.json', dict_infos)
