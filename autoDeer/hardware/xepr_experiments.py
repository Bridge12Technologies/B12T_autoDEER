import importlib
import time
import numpy as np
import re

MODULE_DIR = importlib.util.find_spec('autoDeer').submodule_search_locations[0]

def run_general(api,ps_file:tuple,exp:tuple,settings:dict,variables:dict,run:bool=True)->None:


    if len(ps_file) == 1:
        # Assuming that both the EXP file and the DEF file have the same name bar-extention
        exp_file = MODULE_DIR + ps_file[0] +".exp"
        def_file = MODULE_DIR + ps_file[0] +".def"

    elif len(ps_file) == 2:
        
        # EXP and DEF file have seperate name
        exp_file = MODULE_DIR + ps_file[0] +".exp"
        def_file = MODULE_DIR + ps_file[1] +".def"

    else:
        raise ValueError("ps_file must be of form ['EXP file'] or ['EXP file','DEF file']")

    api.set_PulseSpel_exp_filepath(exp_file)
    api.set_PulseSpel_def_filepath(def_file)
    api.compile_PulseSpel_prg()
    api.compile_PulseSpel_def()   

    if "ReplaceMode" in settings:
        api.set_ReplaceMode(settings["ReplaceMode"]) 
    else:
        api.set_ReplaceMode(False) 
    
    if "PhaseCycle" in settings:
        api.set_PhaseCycle(settings["PhaseCycle"]) 
    else:
        api.set_ReplaceMode(True) 

    if "Acquistion_mode" in settings:
        api.set_Acquistion_mode(settings["Acquistion_mode"])
    else:    
        api.set_Acquistion_mode(1)

    # Identifying a dimension change in settings
    r = re.compile("dim([0-9]*)")
    match_list = list(filter(lambda list: list != None,[r.match(i) for i in settings.keys()]))
    if len(match_list) >= 1:
        for i in range(0,len(match_list)):
            key = match_list[i][0]
            dim = int(r.findall(key)[0])
            new_length = settings[key]
            change_dimensions(exp_file,dim,new_length)
    
    # setting PS Variables

    # Some Defaults first, these are overwritten if needed

    api.set_PulseSpel_var("p0",16)
    api.set_PulseSpel_var("p1",32)


    api.set_PulseSpel_var("d0",400)
    api.set_PulseSpel_var("d1",500)

    api.set_PulseSpel_var("d30",16)
    api.set_PulseSpel_var("d31",16)

    api.set_PulseSpel_var("h",20)
    api.set_PulseSpel_var("n",1000)
    api.set_PulseSpel_var("m",1)

    # Change all variables

    for var in variables:
        api.set_PulseSpel_var(var.lower(),variables[var])

    api.set_PulseSpel_experiment(exp[0])
    api.set_PulseSpel_phase_cycling(exp[1])


    # Compile Defs and Program
    api.compile_PulseSpel_prg()
    api.compile_PulseSpel_def()  

    # Run Experiment
    if run == True:
        api.run_exp()
        time.sleep(1)
    pass

def change_dimensions(path,dim:int,new_length:int):
    """This is a HIGHLY specific function to change  a line in a specific pulse spel file. This will break your pulse spel script if applied to any other file."""
    
    dim_str = "dim" +str(int(dim))
    # Open file
    with open(path, 'r') as file:
        data = file.readlines()

    # Build Regular expression to search for
    re_search = fr"dim{int(dim)}\s*s\[[0-9]+,*[0-9]*\]" # This identifies the correct line and extracts the comment.
    if type(new_length) == int:
        new_string = f"dim{int(dim)} s[{int(new_length)}]"
    elif type(new_length) == list:
        if len(new_length) == 1:
            new_string = f"dim{int(dim)} s[{int(new_length[0])}]"
        elif len(new_length) == 2:
            new_string = f"dim{int(dim)} s[{int(new_length[0])},{int(new_length[1])}]"
        else:
            raise ValueError("New length can't have more than 2 dimensions")

    data = [re.sub(re_search,new_string,line) for line in data]

    with open(path, 'w') as file:
        file.writelines( data )
        
    

def get_nutations(api,nu,field,step,nx:int=128):

    min_freq = nu[0]
    max_freq = nu[1]

    freq_table = np.arange(min_freq,max_freq,step)

    n = len(freq_table)

    if len(field) == 2:
        input_freq = field[0]
        input_field =  field[1]
        start_field = input_field * min_freq / input_freq
    elif len(field) == 1:
        start_field = field
    
    field_table = freq_table * start_field/min_freq
    
    # go to start field /  freq
    api.set_field(field_table[0],hold=True)
    api.set_freq(freq_table[0])

    nut_data = np.zeros((n,nx+1),dtype=np.complex64)

    for i in range(0,n):
        api.set_field(field_table[i],hold=True)
        api.set_freq(freq_table[i])
        api.run_exp()
        while api.is_exp_running():
            time.sleep(0.5)
        
        dataset = api.acquire_dataset()
        nut_data[i,0] = api.get_counterfreq()
        nut_data[i,1:] = dataset.data
        t = dataset.time

    return t,nut_data

