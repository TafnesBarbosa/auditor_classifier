from main import *

def roda_command(director, split_fraction, model):
    os.system(f'python3 main.py -id {director} -sf {split_fraction} -m {model} -mi {50000} -df 2')

def gather_info(director, dict_infos):
    for file in os.listdir(director):
        info = read_info(os.path.join(director, file, 'output_metrics_features.json'))
        dict_infos[file]['psnr_max'].append(info['splatfacto']['psnr_train_max'])
        dict_infos[file]['ssim_max'].append(info['splatfacto']['ssim_train_max'])
        dict_infos[file]['lpips_min'].append(info['splatfacto']['lpips_train_min'])
    return dict_infos

def delete_all(director):
    for file in os.listdir(director):
        os.system(f'rm -rf {os.path.join(director, file, "output_splatfacto")}')
        os.system(f'rm -f {os.path.join(director, file, "output_metrics_features.json")}')
        info = read_info(os.path.join(director, file, "info.json"))
        info['splatfacto'] = {
                "trained": False,
                "evaluations": False
            }
        write_info(os.path.join(director, file, "info.json"), info)

dict_infos = {}
dict_infos['split_fractions'] = []

dict_infos['LAMIA_CORREDOR_BLOCO_L_3_2489-Images'] = {}

dict_infos['LAMIA_CORREDOR_BLOCO_L_3_2489-Images']['psnr_max'] = []

dict_infos['LAMIA_CORREDOR_BLOCO_L_3_2489-Images']['ssim_max'] = []

dict_infos['LAMIA_CORREDOR_BLOCO_L_3_2489-Images']['lpips_min'] = []

if os.path.exists('./tudo_split.json'):
    dict_infos = read_info('./tudo_split.json')

director = "/home/tafnes/Downloads/teste_iuri"

for split_fraction in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    dict_infos['split_fractions'].append(split_fraction)
    delete_all(director)
    roda_command(director, split_fraction, 'splatfacto')
    dict_infos = gather_info(director, dict_infos)
    delete_all(director)
    write_info('./tudo_split.json', dict_infos)
