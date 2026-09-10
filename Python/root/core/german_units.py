from core.enums import ReinforcementType
from core.models import GermanUnit

PZ_LEHR = GermanUnit(ReinforcementType.PZ_DIV, "Panzer Lehr", combat_value=2)

SS_12 = GermanUnit(ReinforcementType.PZ_DIV, "12th SS Panzer", combat_value=2)

SS_1 = GermanUnit(ReinforcementType.PZ_DIV, "1st SS Panzer", combat_value=2)
SS_9 = GermanUnit(ReinforcementType.PZ_DIV, "9th SS Panzer", combat_value=2)
SS_10 = GermanUnit(ReinforcementType.PZ_DIV, "10th SS Panzer", combat_value=2)
SS_2 = GermanUnit(ReinforcementType.PZ_DIV, "2nd SS Panzer", combat_value=2)

TIGER_101 = GermanUnit(ReinforcementType.TIGER_BN, "101st Tiger Battalion", combat_value=3)

PZ_21 = GermanUnit(ReinforcementType.PZ_DIV, "21st Panzer", combat_value=2)
PZ_116 = GermanUnit(ReinforcementType.PZ_DIV, "116th Panzer", combat_value=2)

SS_17_PZGRD = GermanUnit(ReinforcementType.PZ_DIV, "17th SS Panzergrenadier", combat_value=2)

PZ_2 = GermanUnit(ReinforcementType.PZ_DIV, "2nd Panzer", combat_value=2)

PZ_9 = GermanUnit(ReinforcementType.PZ_DIV, "9th Panzer", combat_value=2)

FS_3 = GermanUnit(ReinforcementType.FALLSCHIRMJAGER, "3rd Fallschirmjager", combat_value=2)
FS_5 = GermanUnit(ReinforcementType.FALLSCHIRMJAGER, "5th Fallschirmjager", combat_value=1)

ROMMEL = GermanUnit(ReinforcementType.COMMANDER, "ROMMEL", combat_value=2)
MEYER = GermanUnit(ReinforcementType.COMMANDER, "MEYER", combat_value=1)
MODEL = GermanUnit(ReinforcementType.COMMANDER, "MODEL", combat_value=1)

panzer_divisions_list = [PZ_LEHR, SS_12, SS_1, SS_9, SS_10, SS_2, PZ_21, PZ_116, SS_17_PZGRD, PZ_2, PZ_9]

def create_nebelwerfer():
    return GermanUnit(ReinforcementType.NEBELWERFER, "Nebelwerfer", combat_value=1)


def create_flak88():
    return GermanUnit(ReinforcementType.FLAK_88, "Flak 88", combat_value=2)


def create_kampfgruppe():
    return GermanUnit(ReinforcementType.KAMPFGRUPPE, "Kampfgruppe", combat_value=1)
