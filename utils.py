from csv import DictReader
#from torch import cuda
import subprocess

"""
ISO Language Codes:
Set 1: 2 letter code for a language
Set 3: 3 letter code for a language, including sublanguages of macrolanguage
NLLB Code: set 3 ISO code, underscore, script in which language is written 
"""


logging_on = True


iso_filename = "ISO-conversion.csv"


def load_dict() -> dict:
    """
    
    """
    iso_dict = {}    
    with open(iso_filename, 'r') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            if row['ISO Set 1'] != '':
                set1code = row['ISO Set 1']
                iso_dict[set1code] = row    
    return iso_dict


def get_nllb_code(
        lang: str
        ) -> str:
    """
    
    """
    iso_dict = load_dict()
    nllb_code = iso_dict[lang]['NLLB Code']
    return nllb_code

def get_set3_code(
        lang: str
        ) -> str:
    """
    
    """
    iso_dict = load_dict()
    set3_code = iso_dict[lang]['Set 3 Code']
    return set3_code

def get_giellalt_code(lang):
    iso_dict = load_dict()
    g_code = iso_dict[lang]['giellalt code']
    return g_code


def log_memory_usage():
    if logging_on:
        memory_info = get_rocm_smi()
        print(f"Memory Usage: {memory_info['used']} / {memory_info['total']} MB")
        return
    else:
        return

def get_rocm_smi():
    #print(f"Memory allocated: {cuda.memory_allocated() / 1024 ** 2} MB")
    #print(f"Memory reserved: {cuda.memory_reserved() / 1024 ** 2} MB")
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    parsed = parse_rocm_smi(output)
    return parsed


def parse_rocm_smi(output):
    memory_info = {}
    for line in output.splitlines():
        if 'Used' in line and 'VRAM' in line:
            memory_used = line.split(':')[-1].strip()
            memory_info['used'] = int(int(memory_used) / 1024 ** 2)
        elif 'Total' in line and 'VRAM' in line:
            memory_total = line.split(':')[-1].strip()
            memory_info['total'] = int(int(memory_total) / 1024 ** 2)
    return memory_info