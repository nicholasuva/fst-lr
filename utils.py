from csv import DictReader


"""
ISO Language Codes:
Set 1: 2 letter code for a language
Set 3: 3 letter code for a language, including sublanguages of macrolanguage
NLLB Code: set 3 ISO code, underscore, script in which language is written 
"""



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
