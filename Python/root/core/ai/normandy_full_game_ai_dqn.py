
from random import shuffle, randint
from core.actions.actions_menu import common_pre_action_phase
from core.ai.dqn.dqn_action_adapter import get_action_phase_decision, get_hitler_intervention_decision
from core.ai.headless_action_masks import execute_flat_move_other_units_from_strategic_reserve_ai
from core.allied_advances_phase import do_allied_advances_phase
from core.allied_armies import (
    BRITISH_SECOND_ARMY,
    CANADIAN_FIRST_ARMY,
    US_FIRST_ARMY,
    US_THIRD_ARMY,
    US_VIII_CORPS,
    US_XV_CORPS,
)
from core.game_constants import CYAN, GREEN, RED, RESET
from core.tables.weather import get_weather_result
from core.resources import (
    do_event,
    do_resource_phase_adjustments,
    do_resource_phase_drms,
    do_resource_phase_reinforcements,
)
from core.map.map_model import hitler_approval_track
from core.tables.carpet_bombing import get_carpet_bombing_result, ATTACK_CANCELLED
from core.global_game_state import GlobalGameState

from core.ai.headless_env_manager import (
    configure_automated_inputs,
)

from core.card_utilities import (
        calculate_attack_modifiers,
        get_all_defending_armies,
        get_armies_as_objects,
        calculate_defense_modifiers,
)


def advance_to_next_decision(draw_deck, mid_deck, late_deck):
    configure_automated_inputs(GlobalGameState.headless)
    GAME_WON = "won"
    GAME_LOST = "lost"
    GAME_CONTINUES = None
    card_28_draw_position = None
    us_3_activation_position = None
    us_3_activation_location = None

    def print_attack_strengths(card, weather, carpet_bombing):
        print(CYAN)
        print("========================================")
        print("ATTACK STRENGTHS (ALLIED ARMIES)")
        print("========================================")

        armies = get_armies_as_objects(card)

        if not armies:
            print("\t=>NO ARMIES ATTACKING")
            print(RESET)
            return

        for army in armies:
            calculate_attack_modifiers(
                card=card,
                army=army,
                num_jabos=weather.available_jabos,
                carpet_bombing=carpet_bombing,
                print_modifiers=True,
            )

        print(RESET)

    def check_for_game_end():
        if hitler_approval_track.value == -2:
            print()
            print("========================================")
            print("HITLER APPROVAL HAS FALLEN TO -2")
            print("YOU ARE RELIEVED OF COMMAND")
            print()
            print("YOU LOSE!")
            print("========================================")
            return GAME_LOST

        allied_formations = [
            US_FIRST_ARMY,
            BRITISH_SECOND_ARMY,
            CANADIAN_FIRST_ARMY,
            US_THIRD_ARMY,
            US_VIII_CORPS,
            US_XV_CORPS,
        ]

        if any(
            formation.location
            and formation.location.name == "FALAISE GAP"
            for formation in allied_formations
        ):
            print()
            print("========================================")
            print("ALLIED FORMATION HAS REACHED FALAISE GAP")
            print()
            print("YOU LOSE!")
            print("========================================")
            return GAME_LOST

        if GlobalGameState.cards_drawn == 48:
            return GAME_WON

        return GAME_CONTINUES

    def print_defense_strengths(card, weather):
        print(CYAN)
        print("========================================")
        print("DEFENSE STRENGTHS (ALLIED ARMIES)")
        print("========================================")

        if GlobalGameState.cards_drawn == 0:
            print()
            print("N/A")
            print()

        armies = get_all_defending_armies()

        for army in armies:
            calculate_defense_modifiers(
                card=card,
                army=army,
                weather=weather,
                print_modifiers=True,
            )

    while True:
        print()
        print("=========================================")
        print("NORMANDY SOLITAIRE! D-Day to Falaise 1944")
        print("=========================================")
        print()

        game_result = check_for_game_end()

        if game_result == GAME_LOST:
            break

        if GlobalGameState.drawn_cards:
            drawn_ids = [
                str(card.card_id)
                for card in GlobalGameState.drawn_cards
            ]

            print(f"List of Drawn Cards: ({', '.join(drawn_ids)})")
        else:
            print("List of Drawn Cards: []")

        print(f"Cards Drawn: {len(GlobalGameState.drawn_cards)}")

        cards_remaining = 48 - len(GlobalGameState.drawn_cards)

        print(f"Cards Remaining: {cards_remaining}")
        print(f"Cards Remaining in Deck: {len(draw_deck)}")

        print()

        if GlobalGameState.current_card:
            print(f"Current Card: {GlobalGameState.current_card.card_id}")

        if GlobalGameState.current_weather:
            print(
                f"Current Weather: "
                f"{GlobalGameState.current_weather.weather_type.value}"
            )

        print()

        menu_items = [
            "Draw Card",
            "Roll For Weather",
            "Resources Phase",
            "Deploy Non Panzer Div Reinforcements",
            "Allied Advances Phase",
            "Action Phase",
        ]

        for index, text in enumerate(menu_items, start=1):
            if GlobalGameState.current_step == index:
                print(f"{GREEN}> {text}{RESET}")
            else:
                print(f"{RED}  {text}{RESET}")

        print()
        print("Press ENTER to perform next action")
        print("Press G to see game summary")

        can_save_game = (
            GlobalGameState.current_step == 1
            and GlobalGameState.current_card is not None
        )

        can_load_game = GlobalGameState.current_step == 1

        if can_save_game:
            print("Press S to save game")

        if can_load_game:
            print("Press L to load saved game")
            print("Press Q to quit")
            print()

        if GlobalGameState.current_step == 1:
            if cards_remaining == 0:
                print()
                print("DRAW DECK EMPTY")
                break

            drawn_card = draw_deck[0]

            GlobalGameState.current_card = drawn_card
            GlobalGameState.current_weather = None
            GlobalGameState.current_carpet_bombing = 0

            draw_deck.remove(drawn_card)
            GlobalGameState.drawn_cards.append(drawn_card)

            # Record where Card 28 appeared
            if drawn_card.card_id == 28:
                card_28_draw_position = len(GlobalGameState.drawn_cards)

            if (
                drawn_card.card_id == 20
                and not GlobalGameState.mid_deck_added
            ):
                draw_deck.extend(mid_deck)
                shuffle(draw_deck)

                card_28 = next(card for card in draw_deck if card.card_id == 28)
                draw_deck.remove(card_28)
                draw_deck.insert(0, card_28)

                GlobalGameState.mid_deck_added = True
                print(CYAN)
                print()
                print("========================================")
                print("MID DECK ADDED")
                print("DRAW DECK SHUFFLED")
                print("========================================")
                print(RESET)

            if (
                drawn_card.card_id == 37
                and not GlobalGameState.late_deck_added
            ):
                draw_deck.extend(late_deck)
                shuffle(draw_deck)
                GlobalGameState.late_deck_added = True

                print(CYAN)
                print()
                print("========================================")
                print("LATE DECK ADDED")
                print("DRAW DECK SHUFFLED")
                print("========================================")
                print(RESET)

            print(CYAN)
            print()
            print("========================================")
            print(f"DREW CARD {drawn_card.card_id}")
            print("========================================")

            drawn_card.summary()

            print(RESET)
            input("Press ENTER to continue...")

            GlobalGameState.current_step = 2
            continue

        if GlobalGameState.current_step == 2:
            if GlobalGameState.current_card.card_id == 13:
                weather = get_weather_result(1)
                weather_roll = "N/A"
            else:
                weather_roll = randint(1, 6)
                weather = get_weather_result(weather_roll)

            GlobalGameState.current_weather = weather
            GlobalGameState.current_carpet_bombing = 0

            print(CYAN)
            print()
            print("========================================")
            print("WEATHER ROLL")
            print("========================================")
            print()

            print(f"ROLL: {weather_roll}")
            print(f"RESULT: {weather.weather_type.value}")

            if (
                GlobalGameState.current_weather.available_jabos > 0
                and GlobalGameState.current_card.air_power.has_carpet_bombing()
            ):
                print()
                print("========================================")
                print("CARPET BOMBING")
                print("========================================")
                print()

                carpet_roll = randint(1, 6)

                carpet_result = get_carpet_bombing_result(
                    die_roll=carpet_roll,
                    drm=GlobalGameState.current_weather.carpet_bombing_drm,
                )

                print(f"ROLL: {carpet_roll}")

                if GlobalGameState.current_weather.carpet_bombing_drm == 1:
                    print("DRM: +1")
                else:
                    print("DRM: 0")

                if carpet_result.attack_modifier == ATTACK_CANCELLED:
                    print("RESULT: ATTACK CANCELLED")
                    GlobalGameState.current_carpet_bombing = 0
                else:
                    GlobalGameState.current_carpet_bombing = (
                        carpet_result.attack_modifier
                    )

                    print(
                        f"RESULT: "
                        f"{GlobalGameState.current_carpet_bombing:+} "
                        f"ATTACK STRENGTH"
                    )

            print(RESET)
            input("Press ENTER to continue...")

            GlobalGameState.current_step = 3
            continue

        if GlobalGameState.current_step == 3:
            do_event(GlobalGameState.current_card)

            do_resource_phase_adjustments(
                GlobalGameState.current_card
            )

            do_resource_phase_drms(
                GlobalGameState.current_weather.weather_type,
                GlobalGameState.current_card,
            )

            print(
                f"RESOURCE PHASE - weather is "
                f"{GlobalGameState.current_weather.weather_type.value}\n"
            )

            print_attack_strengths(
                GlobalGameState.current_card,
                GlobalGameState.current_weather,
                GlobalGameState.current_carpet_bombing,
            )

            print_defense_strengths(
                GlobalGameState.current_card,
                GlobalGameState.current_weather,
            )

            print(CYAN)

            do_resource_phase_reinforcements(
                GlobalGameState.current_card
            )

            print(RESET)
            input("Press ENTER to continue...")

            GlobalGameState.current_step = 4
            continue

        if GlobalGameState.current_step == 4:
            execute_flat_move_other_units_from_strategic_reserve_ai()

            GlobalGameState.current_step = 5
            continue

        if GlobalGameState.current_step == 5:
            # Record whether Third Army was inactive before this Allied phase
            us_3_was_active = GlobalGameState.us_third_army_activated

            print(CYAN)

            do_allied_advances_phase(
                GlobalGameState.current_card,
                GlobalGameState.current_weather,
            )

            print(RESET)

            # Detect the exact turn on which Third Army became active
            if (
                not us_3_was_active
                and GlobalGameState.us_third_army_activated
            ):
                us_3_activation_position = len(
                    GlobalGameState.drawn_cards
                )

                if US_FIRST_ARMY.location is not None:
                    us_3_activation_location = (
                        US_FIRST_ARMY.location.name
                    )

            game_result = check_for_game_end()

            if game_result == GAME_LOST:
                break


            GlobalGameState.current_step = 6
            proceed = common_pre_action_phase(
                GlobalGameState.current_card,
                GlobalGameState.current_weather,
            )
            if not proceed:
                observation, action_mask = get_hitler_intervention_decision(
                    GlobalGameState.current_card
                )
                return {
                    "decision_type": "hitler_intervention",
                    "observation": observation,
                    "action_mask": action_mask,
                }

            continue


        if GlobalGameState.current_step == 6:
            # Advance to the next turn when the Action Phase is finished
            if (
                GlobalGameState.actions_left_this_turn == 0
                and GlobalGameState.reserve_actions == 0
            ):
                GlobalGameState.current_step = 1
                continue

            observation, action_mask = get_action_phase_decision()
            return {
                "decision_type": "action_phase",
                "observation": observation,
                "action_mask": action_mask,
            }

    print()
    print("========================================")
    print("GAME OVER")
    print()

    if game_result == GAME_WON:
        print("You won!")
    else:
        print("You lost!")

    print("========================================")

    return {
        "result": game_result,
        "card_28_draw_position": card_28_draw_position,
        "us_3_activation_position": us_3_activation_position,
        "us_3_activation_location": us_3_activation_location,
    }
