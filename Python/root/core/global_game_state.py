from core.models import Strategy


class GlobalGameState:
    # MODE if monte_carlo certain things disabled (eg auto save)
    # monte_carlo = False
    headless = False

    us_1_front_line = 11
    brit_2_front_line = 7
    can_1_front_line = 7
    us_3_front_line = 8
    us_viii_front_line = 7
    us_xv_front_line = 4

    # RESOURCE ROLL DRMs
    transport_roll_drm = 0
    supply_roll_drm = 0

    # RESOURCE CHECK DRMs
    transport_check_drm = 0
    supply_check_drm = 0
    hitler_approval_check_drm = 0

    # RESOURCE (AUGMENATION) BASE LEVELS
    transport_base_level = 3
    supply_base_level = 3
    hitler_approval_base_level = 3

    # HUMAN/AI TOGGLES
    german_casualty_strategy = Strategy.RANDOM

    actions_left_this_turn = 0

    # CARD/DECK STATE
    cards_drawn = 0
    drawn_cards = []
    mid_deck_added = False
    late_deck_added = False

    current_card = None
    current_weather = None
    current_carpet_bombing = 0
    current_step = 1

    # TRACK ATTACKED ARMIES THIS TURN
    counter_attacked_armies = set()

    # MISC GAME EVENTS
    cherbourg_captured = False
    us_third_army_activated = False
    us_third_army_merged = False
    us_first_army_furthest_advance = 11
    meyer_available = False

    # BOCAGE MODIFIER, -1 ONCE EVENT CARD DRAWN FOR GERMAN DEFENSE ONLY
    bocage_defense_modifier = 0

    # MODEL IN COMMAND
    model_in_command = False

    # HITLER ASSASSINATION
    hitler_assassination = False
    
    # HITLER INTERVENTION
    hitler_intervention_no_effect = False
    
    reserve_actions = 0
    
    armies_upgraded = False


