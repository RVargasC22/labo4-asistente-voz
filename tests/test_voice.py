import sys
sys.path.insert(0, "src")
from voice import fix_brands

KEYS = {"schwarzkopf": "Schwarzkopf", "maxcio": "Maxcio", "decolortime": "Decolor Time", "salomon": "SALOMON"}


def test_fixes_split_brand_keeping_punctuation():
    assert fix_brands("Busco Schwarz Cops, tinte sin amoniaco", KEYS) == "Busco Schwarzkopf, tinte sin amoniaco"


def test_leaves_ordinary_words_and_exact_brands_alone():
    for s in ("Quiero dos de patinete, máximo 156 euros", "Prefiero que sea de color Plata", "Hola, busco botas salomon. Quiero 2"):
        assert fix_brands(s, KEYS) == s


def test_spoken_text_converts_prices_to_words_in_code():
    from voice import spoken_text
    assert spoken_text("El total es 98,64 euros.") == "El total es noventa y ocho euros con sesenta y cuatro céntimos."
    assert spoken_text("Son 187.5 euros") == "Son ciento ochenta y siete euros con cincuenta céntimos"
    assert spoken_text("Cuesta 1 euro") == "Cuesta un euro"
    assert spoken_text("Cuesta 21,01€") == "Cuesta veintiún euros con un céntimo"
    assert spoken_text("Total 1.446,50 euros") == "Total mil cuatrocientos cuarenta y seis euros con cincuenta céntimos"
    assert spoken_text("3 unidades, 12 euros") == "3 unidades, doce euros"
