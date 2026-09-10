# core/armies.py

from core.models import AlliedArmy
from core.enums import Nation

BRITISH_SECOND_ARMY = AlliedArmy(name="BRITISH XXX CORPS",
                                 nation=Nation.BRIT_2,
                                 reverse_name="BRITISH 2nd ARMY",
                                 _strength=2,
                                 reverse_strength=4)

CANADIAN_FIRST_ARMY = AlliedArmy(name="BRITISH I CORPS",
                                 nation=Nation.CAN_1,
                                 reverse_name="CANADIAN 1st ARMY",
                                 _strength=2,
                                 reverse_strength=4)

US_FIRST_ARMY = AlliedArmy(name="US 1st ARMY",
                           nation=Nation.US_1,
                           reverse_name="US 1st ARMY",
                           _strength=3,
                           reverse_strength=4)

US_VIII_CORPS = AlliedArmy(name="US VIII CORPS",
                           nation=Nation.US_VIII,
                           _strength=2)

US_XV_CORPS = AlliedArmy(name="US XV CORPS", nation=Nation.US_XV, _strength=2)

US_THIRD_ARMY = AlliedArmy(name="US 3rd ARMY", nation=Nation.US_3, _strength=4, merged=True)

armies_list = [US_FIRST_ARMY, BRITISH_SECOND_ARMY, CANADIAN_FIRST_ARMY, US_THIRD_ARMY, US_VIII_CORPS, US_XV_CORPS]
