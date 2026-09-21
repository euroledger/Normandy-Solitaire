import traceback

from core.ai.ai_diagnostics_helper import save_exception_game, save_winning_game
from core.ai.headless_env_manager import _REAL_PRINT_, save_log_to_disk
import builtins
import copy
from random import shuffle, randint
from collections import Counter

from cards.decks import draw_deck, mid_deck, late_deck
from core.actions.strategic_reserve_actions import (
    do_move_other_unit_from_strategic_reserve,
    get_other_units_in_strategic_reserve,
)
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
from core.save_load_game import load_game, save_game
from core.tables.weather import get_weather_result
from core.resources import (
    do_event,
    do_resource_phase_adjustments,
    do_resource_phase_drms,
    do_resource_phase_reinforcements,
)
from core.tables.carpet_bombing import get_carpet_bombing_result, ATTACK_CANCELLED
from core.map.map_utilities import do_opening_setup
from core.game_summary import print_game_summary
from core.global_game_state import GlobalGameState

from core.ai.headless_env_manager import (
    configure_headless_printing,
    configure_automated_inputs,
)
from game_phases.action_phase import action_phase_fork

import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
# Enforce human visible terminal strings
# Set both to False to play the manual (human) game

start_time = time.perf_counter()

# Save an untouched copy of the exact card lists at application startup
_PRISTINE_DRAW = copy.deepcopy(draw_deck)
_PRISTINE_MID = copy.deepcopy(mid_deck)
_PRISTINE_LATE = copy.deepcopy(late_deck)


def execute_single_game_run(i):
    headless = True
    GlobalGameState.headless = headless
    

    if headless:
        configure_headless_printing(i)

    configure_automated_inputs(headless)

    from core.card_utilities import (
        calculate_attack_modifiers,
        get_all_defending_armies,
        get_armies_as_objects,
        calculate_defense_modifiers,
    )

    # Analysis values for this game
    card_28_draw_position = None
    us_3_activation_position = None
    us_3_activation_location = None

    # --- DECK REFRESHMENT FIX ---
    # --- DECK REFRESHMENT (FIXED FOR REFERENCES) ---

    # 1. Clear out tracking parameters on your master control layout
    GlobalGameState.cards_drawn = 0
    GlobalGameState.drawn_cards = []
    GlobalGameState.current_card = None
    GlobalGameState.current_weather = None
    GlobalGameState.current_step = 1
    GlobalGameState.mid_deck_added = False
    GlobalGameState.late_deck_added = False

    # 2. Slice-assign the pristine card contents back to the active tracking lists
    # This keeps the exact memory reference pointers untouched while resetting the items
    import cards.decks

    draw_deck = list(cards.decks.draw_deck)
    mid_deck = list(cards.decks.mid_deck)
    late_deck = list(cards.decks.late_deck)

    GAME_WON = "won"
    GAME_LOST = "lost"
    GAME_CONTINUES = None

    do_opening_setup()

    opening_cards = draw_deck[:2]
    random_cards = draw_deck[2:]
    shuffle(random_cards)
    draw_deck[:] = opening_cards + random_cards

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

    from core.map.map_model import hitler_approval_track

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

        user_input = input("> ").strip().upper()

        if user_input == "Q":
            print("Goodbye.")
            break

        if user_input == "G":
            print_game_summary()
            print()
            input("Press ENTER to continue...")
            continue

        if user_input == "S" and can_save_game:
            save_game()
            print()
            input("Press ENTER to continue...")
            continue

        if user_input == "L" and can_load_game:
            load_game()
            print()
            input("Press ENTER to continue...")
            continue

        if GlobalGameState.current_step == 1 and user_input == "":
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


            # if (
            #     drawn_card.card_id == 20
            #     and not GlobalGameState.mid_deck_added
            # ):
            #     draw_deck.extend(mid_deck)
            #     shuffle(draw_deck)
            #     GlobalGameState.mid_deck_added = True
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

        if GlobalGameState.current_step == 2 and user_input == "":
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

        if GlobalGameState.current_step == 3 and user_input == "":
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

        if GlobalGameState.current_step == 4 and user_input == "":
            while get_other_units_in_strategic_reserve():
                choice = input(
                    "Deploy a non-Panzer unit from Strategic Reserve? (Y/N): "
                ).strip().lower()

                if choice != "y":
                    break

                deployed = do_move_other_unit_from_strategic_reserve()

                if not deployed:
                    break

            GlobalGameState.current_step = 5
            continue

        if GlobalGameState.current_step == 5 and user_input == "":
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
            continue

        if GlobalGameState.current_step == 6 and user_input == "":
            action_phase_fork()
            continue

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

# ============================================================
# MULTIPROCESS BATCH RUNNER
# ============================================================

NUM_EPISODES = 1000


def _silent_print(*args, **kwargs):
    pass


builtins.print = _silent_print

# Start conservatively.
# Change this later once we know multiprocessing is stable.
MAX_WORKERS = 16


def run_game_worker(game_number):

    game_start = time.perf_counter()

    try:
        game_data = execute_single_game_run(
            game_number
        )

        # Save an occasional game for inspection.


        if game_number > 0 and game_number % 500 == 0:
            save_log_to_disk(
                game_number=game_number
            )
        duration = (
            time.perf_counter()
            - game_start
        )

        if game_data["result"] == "won":
            save_winning_game(
                game_number=game_number,
                game_data=game_data,
            )

        return {
            "game_number": game_number,
            "result": game_data["result"],
            "cards_drawn": GlobalGameState.cards_drawn,
            "duration": duration,
            "card_28_draw_position": (
                game_data["card_28_draw_position"]
            ),
            "us_3_activation_position": (
                game_data["us_3_activation_position"]
            ),
            "us_3_activation_location": (
                game_data["us_3_activation_location"]
            ),
            "error": None,
        }

    except Exception as exception:

        duration = (
            time.perf_counter()
            - game_start
        )

        # Save while we are still inside the worker
        # that owns the failed game's state and log.

        save_exception_game(
            game_number=game_number,
            exception=exception,
        )

        return {
            "game_number": game_number,
            "result": "lost",
            "cards_drawn": GlobalGameState.cards_drawn,
            "duration": duration,
            "card_28_draw_position": None,
            "us_3_activation_position": None,
            "us_3_activation_location": None,
            "error": traceback.format_exc(),
        }
def run_parallel_games(
    num_episodes,
    max_workers=MAX_WORKERS,
):
    
    # Submit all games to a pool of independent Python processes.
    # Results are returned to the parent process as games finish.


    results = []

    batch_start = time.perf_counter()

    _REAL_PRINT_()
    _REAL_PRINT_("=========================================")
    _REAL_PRINT_("       MULTIPROCESS BATCH START")
    _REAL_PRINT_("=========================================")
    _REAL_PRINT_(
        f"Games             : {num_episodes}"
    )
    _REAL_PRINT_(
        f"Worker processes  : {max_workers}"
    )
    _REAL_PRINT_(
        f"Logical CPUs      : {os.cpu_count()}"
    )
    _REAL_PRINT_("=========================================")
    _REAL_PRINT_()

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_game = {
            executor.submit(
                run_game_worker,
                game_number,
            ): game_number
            for game_number in range(num_episodes)
        }

        completed = 0

        for future in as_completed(future_to_game):
            game_number = future_to_game[future]

            try:
                result = future.result()

            except Exception:  # noqa: BLE001
                # This catches failures at the multiprocessing
                # infrastructure level rather than errors inside
                # execute_single_game_run().
                result = {
                    "game_number": game_number,
                    "result": "lost",
                    "cards_drawn": 0,
                    "duration": 0.0,
                    "card_28_draw_position": None,
                    "us_3_activation_position": None,
                    "us_3_activation_location": None,
                    "error": traceback.format_exc(),
                }

            results.append(result)

            completed += 1

            # Progress report every 100 completed games.
            if (
                completed % 100 == 0
                or completed == num_episodes
            ):
                elapsed = (
                    time.perf_counter()
                    - batch_start
                )

                games_per_second = (
                    completed / elapsed
                    if elapsed > 0
                    else 0
                )

                _REAL_PRINT_(
                    f"Completed "
                    f"{completed}/{num_episodes} games "
                    f"({games_per_second:.1f} games/sec)"
                )

    wall_clock_time = (
        time.perf_counter()
        - batch_start
    )

    return results, wall_clock_time


def print_batch_summary(
    results,
    wall_clock_time,
):
    
    # Aggregate results in the parent process.
    # Worker processes never modify these aggregate statistics.
    

    # Make output deterministic even though games finish
    # in an arbitrary order.
    results.sort(
        key=lambda result: result["game_number"]
    )

    actual_attempts = len(results)

    win_count = sum(
        1
        for result in results
        if result["result"] == "won"
    )

    loss_count = sum(
        1
        for result in results
        if result["result"] == "lost"
    )

    cards_drawn_per_game = [
        result["cards_drawn"]
        for result in results
    ]

    game_durations = [
        result["duration"]
        for result in results
    ]

    errors = [
        result
        for result in results
        if result["error"] is not None
    ]

    total_cpu_game_time = sum(
        game_durations
    )

    average_game_time = (
        total_cpu_game_time / actual_attempts
        if actual_attempts > 0
        else 0
    )

    avg_cards_processed = (
        sum(cards_drawn_per_game)
        / len(cards_drawn_per_game)
        if cards_drawn_per_game
        else 0
    )

    win_rate = (
        (win_count / actual_attempts) * 100
        if actual_attempts > 0
        else 0
    )

    games_per_second = (
        actual_attempts / wall_clock_time
        if wall_clock_time > 0
        else 0
    )

    _REAL_PRINT_()
    _REAL_PRINT_("=========================================")
    _REAL_PRINT_("     MULTIPROCESS PERFORMANCE SUMMARY")
    _REAL_PRINT_("=========================================")

    _REAL_PRINT_(
        f"Total Games Attempted : "
        f"{actual_attempts}"
    )

    _REAL_PRINT_(
        f"Total Recorded Wins   : "
        f"{win_count} ({win_rate:.2f}%)"
    )

    _REAL_PRINT_(
        f"Total Recorded Losses : "
        f"{loss_count}"
    )

    _REAL_PRINT_(
        f"Avg Cards Processed   : "
        f"{avg_cards_processed:.2f} / 48"
    )

    _REAL_PRINT_(
        f"Worker Errors         : "
        f"{len(errors)}"
    )

    _REAL_PRINT_("-----------------------------------------")

    _REAL_PRINT_(
        f"Wall Clock Time       : "
        f"{wall_clock_time:.4f} seconds"
    )

    _REAL_PRINT_(
        f"Games Per Second      : "
        f"{games_per_second:.2f}"
    )

    _REAL_PRINT_(
        f"Avg Worker Game Time  : "
        f"{average_game_time:.6f} seconds"
    )

    _REAL_PRINT_(
        f"Total Worker CPU Time : "
        f"{total_cpu_game_time:.4f} seconds"
    )

    _REAL_PRINT_("=========================================")

    # ========================================================
    # ERROR REPORT
    # ========================================================

    if errors:
        _REAL_PRINT_()
        _REAL_PRINT_("=========================================")
        _REAL_PRINT_("              WORKER ERRORS")
        _REAL_PRINT_("=========================================")

        for result in errors:
            _REAL_PRINT_()
            _REAL_PRINT_(
                f"GAME {result['game_number']}"
            )
            _REAL_PRINT_(
                result["error"]
            )

        _REAL_PRINT_("=========================================")

    # ========================================================
    # WINNING GAME ANALYSIS
    # ========================================================

    winning_results = [
        result
        for result in results
        if result["result"] == "won"
    ]

    if winning_results:
        _REAL_PRINT_()
        _REAL_PRINT_("=========================================")
        _REAL_PRINT_("          WINNING GAME ANALYSIS")
        _REAL_PRINT_("=========================================")

        winning_card_28_positions = [
            result["card_28_draw_position"]
            for result in winning_results
            if result["card_28_draw_position"]
            is not None
        ]

        if winning_card_28_positions:
            _REAL_PRINT_(
                f"Card 28 Avg Draw       : "
                f"{sum(winning_card_28_positions) / len(winning_card_28_positions):.2f}"
            )

            _REAL_PRINT_(
                f"Card 28 Earliest       : "
                f"{min(winning_card_28_positions)}"
            )

            _REAL_PRINT_(
                f"Card 28 Latest         : "
                f"{max(winning_card_28_positions)}"
            )

        activated_positions = [
            result["us_3_activation_position"]
            for result in winning_results
            if result["us_3_activation_position"]
            is not None
        ]

        never_activated = sum(
            1
            for result in winning_results
            if result["us_3_activation_position"]
            is None
        )

        if activated_positions:
            _REAL_PRINT_(
                f"US 3rd Army Avg Active : "
                f"{sum(activated_positions) / len(activated_positions):.2f}"
            )

            _REAL_PRINT_(
                f"US 3rd Army Earliest   : "
                f"{min(activated_positions)}"
            )

            _REAL_PRINT_(
                f"US 3rd Army Latest     : "
                f"{max(activated_positions)}"
            )

        _REAL_PRINT_(
            f"US 3rd Never Activated : "
            f"{never_activated}"
        )

        activation_locations = Counter(
            result["us_3_activation_location"]
            for result in winning_results
            if result["us_3_activation_location"]
            is not None
        )

        _REAL_PRINT_()
        _REAL_PRINT_(
            "US 3rd Army Activation Locations:"
        )

        if activation_locations:
            for location, count in (
                activation_locations.items()
            ):
                _REAL_PRINT_(
                    f"  {location}: {count}"
                )
        else:
            _REAL_PRINT_("  NONE")

        _REAL_PRINT_("=========================================")


def main():
    results, wall_clock_time = run_parallel_games(
        num_episodes=NUM_EPISODES,
        max_workers=MAX_WORKERS,
    )

    print_batch_summary(
        results,
        wall_clock_time,
    )


if __name__ == "__main__":
    # Required on Windows when creating child processes.
    multiprocessing.freeze_support()

    main()