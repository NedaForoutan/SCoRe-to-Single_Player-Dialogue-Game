from typing import Dict, List, Union
from string import Template
from dataclasses import dataclass
import random
import copy
import logging

from clemcore import backends
from clemcore.clemgame import GameBenchmark, GameSpec
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.master import GameState, Outcome

from constants import *
from validation_errors import *
from players import ClueGiver, Guesser
from board import CodenamesBoard
from scorer import CodenamesScorer

logger = logging.getLogger(__name__)

@dataclass
class CodenamesGameState(GameState):
    board: CodenamesBoard = None
    opponent_difficulty: int = None
    assassin_won: bool = False
    invalid_response: bool = False
    request_count: int = 0
    parsed_request_count: int = 0
    violated_request_count: int = 0

    def __post_init__(self):
        super().__init__()


class CodenamesGame(DialogueGameMaster):
    """This class implements a codenames game in which player A
    is giving a clue for a set of target words on a board, 
    which player B has to guess from the given clue.
    """

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[backends.Model]):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        board: CodenamesBoard = CodenamesBoard(game_instance[ASSIGNMENTS][TEAM],
                                                    game_instance[ASSIGNMENTS][OPPONENT],
                                                    game_instance[ASSIGNMENTS][INNOCENT],
                                                    game_instance[ASSIGNMENTS][ASSASSIN],
                                                    game_instance[BOARD],
                                                    self.experiment["flags"])
        self.state = CodenamesGameState(
            board = board,
            opponent_difficulty = self.experiment[OPPONENT_DIFFICULTY],
        )

        self.cluegiver: ClueGiver = ClueGiver(self.player_models[0], self.experiment["flags"])
        self.guesser: Guesser = Guesser(self.player_models[1], self.experiment["flags"])
        self.add_player(self.cluegiver)
        self.add_player(self.guesser)

    def _was_target(self, word: str):
        return word in self.cluegiver.targets

    def _get_cluegiver_prompt(self, initial=False) -> str:
        folder = "initial_prompts" if initial else "intermittent_prompts"
        prompt_cluegiver = self.load_template(f"resources/{folder}/prompt_cluegiver")

        team_words = ", ".join(self.state.board.get_hidden_words(TEAM))
        opponent_words = ", ".join(self.state.board.get_hidden_words(OPPONENT))
        innocent_words = ", ".join(self.state.board.get_hidden_words(INNOCENT))
        assassin_words = ", ".join(self.state.board.get_hidden_words(ASSASSIN))

        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words=team_words,
                                                                          opponent_words=opponent_words,
                                                                          innocent_words=innocent_words,
                                                                          assassin_words=assassin_words)
        return instance_prompt_cluegiver

    def _get_guesser_initial_prompt(self):
        return self._get_guesser_prompt("initial_prompts")

    def _get_guesser_intermittent_prompt(self):
        return self._get_guesser_prompt("intermittent_prompts")

    def _get_guesser_prompt(self, folder) -> str:
        prompt_guesser = self.load_template(f"resources/{folder}/prompt_guesser")

        board = ", ".join(self.state.board.get_all_hidden_words())
        instance_prompt_guesser = Template(prompt_guesser).substitute(board=board,
                                                                      clue=self.cluegiver.clue,
                                                                      number=self.cluegiver.number_of_targets)
        return instance_prompt_guesser

    def _on_before_game(self):
        # add initial cluegiver prompt
        self.set_context_for(self.cluegiver, self._get_cluegiver_prompt(True))

    def _on_before_round(self):
        # let mock opponent reveal their cards
        if self.current_round > 0:
            self._opponent_turn()
        self.log_to_self(Turn_logs.BOARD_STATUS, self.state.board.get_current_board())
        self.cluegiver.retries = 0
        self.guesser.retries = 0

    def _validate_player_response(self, player: Union[ClueGiver, Guesser], utterance: str) -> bool:
        self.state.request_count += 1
        self.state.invalid_response = False
        if player == self.cluegiver:
            try:
                player.validate_response(utterance, self.state.board.get_revealed_words(TEAM),
                                         self.state.board.get_all_hidden_words())
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.state.invalid_response = True
                self.state.violated_request_count += 1
                self.last_error_message = error.attributes["message"]
        else:
            try:
                player.validate_response(utterance, self.state.board.get_revealed_words(TEAM),
                                         self.state.board.get_all_hidden_words(), self.cluegiver.number_of_targets,
                                         self.cluegiver.clue)
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.state.invalid_response = True
                self.state.violated_request_count += 1
                self.last_error_message = error.attributes["message"]
        if self.state.invalid_response:
            self.state.abort()
        return not self.state.invalid_response

    def _parse_response(self, player: Union[ClueGiver, Guesser], utterance: str) -> str:
        self.state.parsed_request_count += 1
        if player == self.cluegiver:
            utterance = player.parse_response(utterance, self.state.board.get_all_hidden_words())
            self.log_to_self(Turn_logs.CLUE, player.clue)
            self.log_to_self(Turn_logs.TARGETS, player.targets)
            return utterance
        else:
            parsed_utterance = player.parse_response(utterance, self.state.board.get_all_hidden_words())
            self.log_to_self(Turn_logs.GUESSES, player.guesses)

            return parsed_utterance

    def _should_pass_turn(self):
        player: Union[ClueGiver, Guesser] = self.current_player
        if player.flags["REPROMPT ON ERROR"]:
            if player.retries < MAX_RETRIES:
                if self.state.invalid_response:
                    logger.debug("Reprompting...")
                    player.retries += 1
                    player.flags_engaged["REPROMPT ON ERROR"] += 1
                    self.set_context_for(player,
                                         f"Your answer did not follow the requested format: {self.last_error_message}")
                    return False
        return True

    def _on_valid_player_response(self, player: Union[ClueGiver, Guesser], parsed_response: str):
        if player == self.cluegiver:
            # score cluegiver precision
            for target in player.targets:
                assignment = self.state.board.get_word_assignment(target)
                self.log_to_self(Turn_logs.WORD_TARGETED, {"word": target, "assignment": assignment})
            # add response of cluegiver embedded in guesser prompt to guesser history
            if self.current_round == 0:
                self.set_context_for(self.guesser, self._get_guesser_initial_prompt())
            else:
                context = self.get_context_for(self.guesser)
                context["content"] += f"\n{self._get_guesser_intermittent_prompt()}"
        else:
            evaluated_guesses = []
            # reveal guesses in order
            for guess in player.guesses:
                assignment = self.state.board.reveal_word(guess)
                if not assignment:
                    continue
                evaluated_guesses.append((guess, assignment))
                # TODO: add player messages here, whether word was revealed and correct,
                # or incorrect and all other guesses were ignored
                self.log_to_self(Turn_logs.TEAM_REVEALED, {"word": guess, "assignment": assignment})
                if self._was_target(guess):
                    self.log_to_self(Turn_logs.TARGET_REVEALED, {"word": guess, "assignment": assignment})
                if not self.state.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break

            guess_feedback = ""
            if evaluated_guesses[-1][1] == TEAM:
                if len(evaluated_guesses) >= 2:
                    guess_feedback = (f"The words {', '.join([guess for guess, assignment in evaluated_guesses])} "
                                      f"were guessed correctly. ")
                else:
                    guess_feedback = f"The word {evaluated_guesses[0][0]} was guessed correctly. "
            else:
                correct_guesses = evaluated_guesses[0:-1]
                incorrect_guess = evaluated_guesses[-1]
                if len(correct_guesses) >= 2:
                    guess_feedback += (
                        f"The words {', '.join([guess for guess, assignment in correct_guesses])} "
                        f"were guessed correctly. ")
                elif len(correct_guesses) == 1:
                    guess_feedback += f"The word {correct_guesses[0][0]} was guessed correctly. "
                guess_feedback += f"The word {incorrect_guess[0]} was guessed but is an {incorrect_guess[1]} word. "

            cluegiver_guess_feedback = copy.copy(guess_feedback)
            cluegiver_guess_feedback += "Your teammate's turn ended there."

            guesser_guess_feedback = copy.copy(guess_feedback)
            guesser_guess_feedback += "Your turn ended there."

            # add guess feedback to guesser history
            self.set_context_for(self.guesser, guesser_guess_feedback)

            # add guesser utterance to cluegiver history and new cluegiver prompt
            self.set_context_for(self.cluegiver, f"{cluegiver_guess_feedback}\n{self._get_cluegiver_prompt(False)}")
        # check if winning/lose conditions are met
        if self.state.board.has_team_won():
            self.state.succeed()
            self.state.assassin_won = False
            self.log_to_self("game end", "team has won")
        elif self.state.board.has_opponent_won():
            self.state.failed()
            self.state.assassin_won = False
            self.log_to_self("game end", "opponent has won")
        elif self.state.board.has_team_won_through_assassin():
            self.state.succeed()
            self.state.assassin_won = True
            self.log_to_self("game end", "team has won through assassin")
        elif self.state.board.has_opponent_won_through_assassin():
            self.state.failed()
            self.state.assassin_won = True
            self.log_to_self("game end", "opponent has won through assassin")

    def _opponent_turn(self):
        # reveal as many opponent cards as the opponent difficulty
        hidden_opponent_words = self.state.board.get_hidden_words(OPPONENT)
        opponent_words = random.sample(hidden_opponent_words, min(self.state.opponent_difficulty, len(hidden_opponent_words)))
        for word in opponent_words:
            assignment = self.state.board.reveal_word(word, OPPONENT)
            self.log_to_self(Turn_logs.OPPONENT_REVEALED, {"word": word, "assignment": assignment})

    def _on_after_game(self):
        # log everything that is needed for score calculation and game evaluation
        self.log_key(BOARD_END_STATUS, self.state.board.get_current_board())
        self.log_key(NUMBER_OF_TURNS, self.current_round + 1)
        self.log_key(METRIC_ABORTED, self.state.outcome == Outcome.ABORTED)
        self.log_key(METRIC_LOSE, self.state.outcome == Outcome.FAILURE)
        self.log_key(GAME_ENDED_THROUGH_ASSASSIN, self.state.assassin_won)
        # METRIC_SUCCESS does not need to be logged as it is inferred from ABORTED and LOSE
        self.log_key(METRIC_REQUEST_COUNT, self.state.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.state.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.state.violated_request_count)
        self.log_key("Cluegiver engaged flags", self.cluegiver.flags_engaged)
        self.log_key("Guesser engaged flags", self.guesser.flags_engaged)


class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)
        random.seed(SEED)

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> DialogueGameMaster:
        return CodenamesGame(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment_config, game_instance) -> GameScorer:
        return CodenamesScorer(self.game_name, experiment_config, game_instance)
