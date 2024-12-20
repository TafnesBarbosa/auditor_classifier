from main import *

def roda_command(director, split_fraction):
    os.system(f'python3 main.py -vd {director} -sf {split_fraction}')

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

parseri = argparse.ArgumentParser(description="Script with argparse options")

# Add arguments
parseri.add_argument("-v", "--director", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)

# Parse arguments
argis = parseri.parse_args()

dict_infos = {}
dict_infos['split_fractions'] = []

dict_infos['interno_vertical_30_natural_normal'] = {}
dict_infos['interno_vertical_60_natural_normal'] = {}
dict_infos['interno_vertical_120_natural_normal'] = {}
dict_infos['externo_horizontal_30_sol_chao-normal'] = {}

dict_infos['interno_vertical_30_natural_normal']['psnr_max'] = []
dict_infos['interno_vertical_60_natural_normal']['psnr_max'] = []
dict_infos['interno_vertical_120_natural_normal']['psnr_max'] = []
dict_infos['externo_horizontal_30_sol_chao-normal']['psnr_max'] = []

dict_infos['interno_vertical_30_natural_normal']['ssim_max'] = []
dict_infos['interno_vertical_60_natural_normal']['ssim_max'] = []
dict_infos['interno_vertical_120_natural_normal']['ssim_max'] = []
dict_infos['externo_horizontal_30_sol_chao-normal']['ssim_max'] = []

dict_infos['interno_vertical_30_natural_normal']['lpips_min'] = []
dict_infos['interno_vertical_60_natural_normal']['lpips_min'] = []
dict_infos['interno_vertical_120_natural_normal']['lpips_min'] = []
dict_infos['externo_horizontal_30_sol_chao-normal']['lpips_min'] = []

if os.path.exists('./tudo_split.json'):
    dict_infos = read_info('./tudo_split.json')

director = "/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Teste_database"

for split_fraction in [0.9, 0.7, 0.5, 0.3, 0.1]:
    dict_infos['split_fractions'].append(split_fraction)
    # delete_all(argis.director)
    roda_command(argis.director, split_fraction)
    dict_infos = gather_info(argis.director, dict_infos)
    delete_all(argis.director)
    # delete_all(director)
    # roda_command(director, split_fraction)
    # dict_infos = gather_info(director, dict_infos)
    # delete_all(director)
    write_info('./tudo_split.json', dict_infos)
