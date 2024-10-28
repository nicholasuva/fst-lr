from csv import DictReader
#from torch import cuda
import subprocess

import pandas as pd

"""
ISO Language Codes:
Set 1: 2 letter code for a language
Set 3: 3 letter code for a language, including sublanguages of macrolanguage
NLLB Code: set 3 ISO code, underscore, script in which language is written 
"""


logging_on = True


iso_filename = "ISO-conversion.csv"

#to deprecate
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

#new, coing to deprecate though probably, mmmmmm maybe not?
def load_lang_codes(filename):
    """
    loads the csv file containing all language codes for all giellaLT languages.
    returns a dict of all the languages and their corresponding codes, with giellaLT code as the key
    giellaLT code must be used as the key because there are wait
    actually i will use iso set 3 as the key

    """
    iso_dict = {}
    with open(filename, 'r') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            if row['ISO Set 3'] != '':
                set_3 = row['ISO Set 3']
                if set_3 not in iso_dict:
                    iso_dict[set_3] = row
                    giellalt_code = iso_dict[set_3]['giellalt code']
                    iso_dict[set_3]['giellalt code'] = [giellalt_code]
                else:
                    #if there are multiple giellalt repos for different versions of tools for the same language
                    #then add the new giellalt code to the list
                    giellalt_code = row['giellalt code']
                    iso_dict[set_3]['giellalt code'].append(giellalt_code)
    return iso_dict


def lang_code_query(
        input_code, #must be iso 3
        output_type
):
    """
    I think what I will have this thingy do is this,
    take in the iso set 3 code, and only that
    return the entire dict for that code
    then you can maybe like just do it all downstream
    """
    iso_dict = load_lang_codes('ISO-conversion.csv')
    try:
        output_code = iso_dict[input_code][output_type]
    except:
        output_code = 'error'
    return output_code


#to deprecate
def get_nllb_code(
        lang: str
        ) -> str:
    """
    
    """
    iso_dict = load_dict()
    nllb_code = iso_dict[lang]['NLLB Code']
    return nllb_code

#to deprecate
def get_set3_code(
        lang: str
        ) -> str:
    """
    
    """
    if len(lang) == 2:
        iso_dict = load_dict()
        set3_code = iso_dict[lang]['ISO Set 3']
        return set3_code
    elif len(lang) == 3:
        return lang

#to deprecate
def get_giellalt_code(lang):
    if len(lang) == 2:
        iso_dict = load_dict()
        g_code = iso_dict[lang]['giellalt code']
        return g_code
    elif len(lang) == 3:
        return lang


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