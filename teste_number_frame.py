from main import *

def roda_command(director, frame_number):
    os.system(f'python3 main.py -vd {director} -fn {frame_number}')

def gather_info(director, dict_infos):
    for file in os.listdir(director):
        info = read_info(os.path.join(director, file, 'output_metrics_features.json'))
        dict_infos[file]['percentage_poses_found'].append(info['percentage_poses_found'])
        dict_infos[file]['num_frames'].append(len(os.listdir(os.path.join(director, file, 'images'))))
        dict_infos[file]['num_tries'].append(info['number_iterations_colmap'])
    return dict_infos

def delete_all(director):
    for file in os.listdir(director):
        os.system(f'rm -rf {os.path.join(director, file, "colmap")}')
        os.system(f'rm -rf {os.path.join(director, file, "images")}')
        os.system(f'rm -rf {os.path.join(director, file, "images_2")}')
        os.system(f'rm -rf {os.path.join(director, file, "images_4")}')
        os.system(f'rm -rf {os.path.join(director, file, "images_8")}')
        os.system(f'rm -f {os.path.join(director, file, "info.json")}')
        os.system(f'rm -f {os.path.join(director, file, "output_metrics_features.json")}')
        os.system(f'rm -f {os.path.join(director, file, "sparse_pc.ply")}')
        os.system(f'rm -f {os.path.join(director, file, "transforms.json")}')

parseri = argparse.ArgumentParser(description="Script with argparse options")

# Add arguments
parseri.add_argument("-v", "--director", type=str, help="Folder with videos. Do not use ./ to refer to the folder. Use the absolute path.", default=None)

# Parse arguments
argis = parseri.parse_args()

dict_infos = {}
dict_infos['frame_numbers'] = [*range(60,30,-2)]

dict_infos['interno_vertical_30_natural_normal'] = {}
dict_infos['interno_vertical_60_natural_normal'] = {}
dict_infos['interno_vertical_120_natural_normal'] = {}
dict_infos['externo_horizontal_30_sol_chao-normal'] = {}

dict_infos['interno_vertical_30_natural_normal']['percentage_poses_found'] = []
dict_infos['interno_vertical_60_natural_normal']['percentage_poses_found'] = []
dict_infos['interno_vertical_120_natural_normal']['percentage_poses_found'] = []
dict_infos['externo_horizontal_30_sol_chao-normal']['percentage_poses_found'] = []

dict_infos['interno_vertical_30_natural_normal']['num_frames'] = []
dict_infos['interno_vertical_60_natural_normal']['num_frames'] = []
dict_infos['interno_vertical_120_natural_normal']['num_frames'] = []
dict_infos['externo_horizontal_30_sol_chao-normal']['num_frames'] = []

dict_infos['interno_vertical_30_natural_normal']['num_tries'] = []
dict_infos['interno_vertical_60_natural_normal']['num_tries'] = []
dict_infos['interno_vertical_120_natural_normal']['num_tries'] = []
dict_infos['externo_horizontal_30_sol_chao-normal']['num_tries'] = []

if os.path.exists('./tudo.json'):
    dict_infos = read_info('./tudo.json')
    k = len(dict_infos['externo_horizontal_30_sol_chao-normal']['num_frames'])
else:
    k = 0

director = "/media/tafnes/0E94B37D94B365BD/Users/tafne/Documents/Teste_database"

for frame_number in range(60-2*k,30,-2):
    dict_infos['frame_numbers'].append(frame_number)
    # delete_all(argis.director)
    # roda_command(argis.director, frame_number)
    # dict_infos = gather_info(argis.director, dict_infos)
    # delete_all(argis.director)
    delete_all(director)
    roda_command(director, frame_number)
    dict_infos = gather_info(director, dict_infos)
    # delete_all(director)
    # print_progress_bar((20-frame_number/15), 19)
    write_info('./tudo.json', dict_infos)


    