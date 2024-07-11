#rip the data from the madlad paper and sort the languages

all_giellalt_langs = [('aka', 'Akan'),
                      ('ale', 'Aleut'),
                      ('amh', 'Amharic'),
                      ('apu', 'Apurinã'),
                      ('ara', 'Arabic'),
                      ('aym', 'Aymara'),
                      ('bak', 'Bashkir'),
                      #('bla', 'Siksika'), codec error reading file
                      ('bul', 'Bulgarian'),
                      ('bxr', 'Russia Buriat'),
                      ('ces', 'Czech'),
                      ('cho', 'Choctaw'),
                      ('chp', 'Chipewyan'),
                      ('chr', 'Cherokee'),
                      ('ciw', 'Chippewa'),
                      ('ckt', 'Chukot'),
                      ('cor', 'Cornish'),
                      ('crj', 'Southern East Cree'),
                      ('crk', 'Plains Cree'),
                      ('cwd', 'Woods Cree'),
                      ('dag', 'Dagbani'),
                      ('deu', 'German'),
                      ('dgr', 'Dogrib'),
                      ('dsb', 'Lower Sorbian'),
                      ('eng', 'English'),
                      ('epo', 'Esperanto'),
                      ('ess', 'Central Siberian Yupik'),
                      ('est-x-plamk', 'Estonian (plamk)'),
                      ('est-x-utee', 'Estonian (utee)'),
                      ('esu', 'Central Alaskan Yup\'ik'),
                      ('eus', 'Basque'),
                      ('evn', 'Evenki'),
                      ('fao', 'Faroese'),
                      ('fin', 'Finnish'),
                      ('fit', 'Meänkieli (Tornedalen Finnish)'),
                      ('fkv', 'Kven Finnish'),
                      ('fro', 'Old French (842-ca. 1400)'),
                      ('gle', 'Irish'),
                      ('got', 'Gothic'),
                      ('grn', 'Guarani'),
                      ('gur', 'Farefare'),
                      ('hdn', 'Northern Haida'),
                      ('hil', 'Hiligaynon'),
                      ('hin', 'Hindi'),
                      ('hun', 'Hungarian'),
                      ('iku', 'Inuktitut'),
                      ('inp', 'Iñapari'),
                      ('ipk', 'Inupiaq'),
                      ('izh', 'Ingrian'),
                      ('kal', 'Kalaallisut'),
                      ('kca', 'Khanty'),
                      ('kek', 'Qʼeqchiʼ'),
                      ('khk', 'Halh Mongolian'),
                      ('kio', 'Kiowa'),
                      ('kjh', 'Khakas'),
                      ('kmr', 'Northern Kurdish'),
                      ('koi', 'Komi-Permyak'),
                      ('kpv', 'Komi-Zyrian'),
                      ('krl', 'Karelian'),
                      ('lav', 'Latvian'),
                      ('lit', 'Lithuanian'),
                      ('liv', 'Liv'),
                      ('luo', 'Luo (Kenya and Tanzania)'),
                      ('lut', 'Lushootseed'),
                      ('mdf', 'Moksha'),
                      ('mhr', 'Eastern Mari'),
                      ('mns', 'Mansi'),
                      ('moh', 'Mohawk'),
                      ('mpj', 'Wangkajunga'),
                      ('mrj', 'Western Mari'),
                      ('mya', 'Burmese'),
                      ('myv', 'Erzya'),
                      ('ndl', 'Ndolo'),
                      ('nds', 'Low German'),
                      ('nio', 'Nganasan'),
                      ('nno', 'Norwegian Nynorsk'),
                      ('nno-x-ext-apertium', 'Norwegian Nynorsk (Apertium)'),
                      ('nob', 'Norwegian Bokmål'),
                      ('non', 'Old Norse'),
                      ('nso', 'Pedi'),
                      ('oji', 'Ojibwa'),
                      ('olo', 'Livvi'),
                      ('pma', 'Paama'),
                      ('quc-x-ext-apertium', 'K\'iche\' (Apertium)'),
                      ('qya', 'Quenya'),
                      ('rmf', 'Kalo Finnish Romani'),
                      ('rmg', 'Traveller Norwegian'),
                      ('rmu-x-testing', 'Tavringer Romani'),
                      ('rmy', 'Vlax Romani'),
                      ('ron', 'Romanian'),
                      ('rue', 'Rusyn'),
                      ('rup', 'Macedo-Romanian/Aromanian'),
                      ('rus', 'Russian'),
                      ('sel', 'Selkup'),
                      ('sjd', 'Kildin Sami'),
                      ('sje', 'Pite Sami'),
                      ('sjt', 'Ter Sami'),
                      ('sju-x-sydlapsk', '18th century Southern Saami'),
                      ('skf', 'Sakirabiá'),
                      ('sma', 'Southern Sami'),
                      ('sme', 'Northern Sami'),
                      ('smj', 'Lule Sámi'),
                      ('smn', 'Inari Sami'),
                      ('sms', 'Skolt Sami'),
                      ('som', 'Somali'),
                      ('spa-x-ext-apertium', 'Spanish'),
                      ('sqi', 'Albanian'),
                      ('srs', 'Tsuut\'ina (Sarsi)'),
                      ('sto', 'Stoney'),
                      ('swe', 'Swedish'),
                      ('tat', 'Tatar'),
                      ('tau', 'Upper Tanana'),
                      ('tel', 'Telugu'),
                      ('tgl', 'Tagalog'),
                      ('tha', 'Thai'),
                      ('tir', 'Tigrinya'),
                      ('tku', 'Upper Necaxa Totonac'),
                      ('tlh', 'Klingon'),
                      ('tqn', 'Sahaptin Tenino'),
                      ('tur-x-ext-trmorph', 'Turkish'),
                      ('tuv', 'Turkana'),
                      ('tyv', 'Tuvinian'),
                      ('udm', 'Udmurt'),
                      ('vep', 'Veps'),
                      ('vot', 'Votic'),
                      ('vot-x-ext-kkankain', 'Votic (kkankain)'),
                      ('vro', 'Võro'),
                      ('xak', 'Maku'),
                      ('xal', 'Kalmyk'),
                      ('xin-x-qda', 'Guazacapán'),
                      ('xwo', 'Written Oirat'),
                      ('yrk', 'Nenets'),
                      ('zul-x-exp', 'Zulu'),
                      ('zxx', 'No linguistic content')
                      ]


glt_codes, glt_langs = zip(*all_giellalt_langs)

glt_sorted = sorted(glt_langs)
print(glt_sorted)
#glt_langs_set = set(glt_langs)

#print(glt_langs_set)

#the data will be in lines
def read_file(filename):
    lines = []
    with open(filename, 'r') as source:
        for line in source:
            lines.append(line.rstrip('\n'))
    return lines

def lines_split(line):
    this_lang_info = {}
    tokens = line.split(' ')
    #print(tokens)
    this_lang_info['code'] = tokens[0]
    this_lang_info['lang'] = ' '.join([tok for tok in tokens[1:-7]])
    this_lang_info['script'] = tokens[-7]
    this_lang_info['docs noisy'] = tokens[-6]
    this_lang_info['docs clean'] = tokens[-5]
    this_lang_info['sents noisy'] = tokens[-4]
    this_lang_info['sents clean'] = tokens[-3]
    this_lang_info['chars noisy'] = tokens[-2]
    this_lang_info['chars clean'] = tokens[-1]
    #print(this_lang_info)
    return this_lang_info

def main():
    filename = 'madlad languages.txt'
    lines = read_file(filename)
    madlad_langs = []
    for line in lines:
        info = lines_split(line)
        madlad_langs.append(info['lang'])
    

    madlad_sorted = sorted(madlad_langs)

    print('langs in both lists:\n')
    for lang in madlad_sorted:
        if lang in glt_sorted:
            print(lang)


    return

main()