git clone "https://github.com/giellalt/lang-fin.git"


hfst-lexc -v "lang-fin/src/fst/morphology/root.lexc" "lang-fin/src/fst/morphology/stems/nouns.lexc" "lang-fin/src/fst/morphology/affixes/nouns.lexc" | hfst-fst2txt | hfst-txt2fst -o lang-fin/src/fst/morphology/nouns-lex.hfst

hfst-twolc -v lang-fin/src/fst/morphology/phonology.twolc -o lang-fin/src/fst/morphology/nouns-twol.hfst

hfst-compose-intersect -v lang-fin/src/fst/morphology/nouns-lex.hfst lang-fin/src/fst/morphology/nouns-twol.hfst | hfst-fst2txt | hfst-txt2fst -o lang-fin/src/fst/morphology/nouns-fin.hfst

hfst-fst2strings -v lang-fin/src/fst/morphology/nouns-fin.hfst -o "nouns-fin-corpus.txt" -c 0