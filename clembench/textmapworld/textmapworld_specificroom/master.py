from clemcore.backends import Model, CustomResponseModel
from clemcore.clemgame import GameMaster, GameBenchmark, Player, GameSpec
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from clemcore.clemgame.master import GameState
from clemcore.utils import file_utils, string_utils

from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import numpy as np
import ast
from queue import Queue
from copy import deepcopy
import re
import random

from textmapworld_utils import loop_identification, get_directions, string_available_directions, have_common_element, \
    get_nextnode_label

INVALID = 0


class PathGuesser(Player):

    def _custom_response(self, context):
        random_path = random.choice(["north", "south", "east", "west"])
        return f'GO: {random_path}'


class PathDescriber(Player):

    def __init__(self, game_instance: Dict):
        super().__init__(CustomResponseModel())
        self.graph_type = game_instance['Game_Type']
        self.ambiguity = game_instance["Ambiguity"]
        self.moves = ast.literal_eval(game_instance['Moves'])
        self.directions = ast.literal_eval(game_instance['Directions'])
        self.move_construction = game_instance["Move_Construction"]
        self.stop_construction = game_instance["Stop_Construction"]
        self.nodes = ast.literal_eval(game_instance['Graph_Nodes'])
        self.edges = ast.literal_eval(game_instance['Graph_Edges'])
        self.positive_answer = game_instance["Player2_positive_answer"]
        self.negative_answer = game_instance["Player2_negative_answer"]
        self.directions_next_node = None
        self.old_node = None
        self.move_type = None
        self.visited_nodes = []
        self.current_node = game_instance["Current_Position"]
        self.visited_nodes.append(self.current_node)

    def check_path_answer(self, utterance: str, directions: List[str], node, saved_node) -> List[Dict]:
        previous_direction = get_directions(node, directions, saved_node)
        previous_dirrection_changed = string_available_directions(previous_direction)
        previous_dirrection_no_pq = string_utils.remove_punctuation(previous_dirrection_changed)
        if not have_common_element(utterance, previous_dirrection_no_pq):
            return [{
                "message": f"The desired direction is not in available paths",
                "type": 0}]

    def validate_answer(self, utterance):
        "Check if the direction is valid"
        the_last_node = self.visited_nodes[-1]
        self.old_node = the_last_node
        errors = self.check_path_answer(utterance, self.directions, the_last_node, self.old_node)
        if errors:
            error = errors[0]
            self.game_error = error
            self.directions_next_node = get_directions(the_last_node, self.directions, the_last_node)
            self.directions_next_node = string_available_directions(self.directions_next_node)
            return "not valid"
        else:
            next_node_label, self.move_type = get_nextnode_label(self.moves, the_last_node, utterance,
                                                                 self.move_construction)
            self.current_node = next_node_label
            if next_node_label in self.nodes:
                self.visited_nodes.append(next_node_label)
                list_directions_nextnode = get_directions(next_node_label, self.directions, self.current_node)
                self.directions_next_node = string_available_directions(list_directions_nextnode)
                return True

    def turn_information(self):
        "Returns the information of the current turn"
        turn_info = {}
        turn_info["from"] = self.old_node
        turn_info["to"] = self.current_node
        return turn_info

    def _custom_response(self, context) -> str:
        "Generate the response for the player"
        utterance = None
        found = re.search(self.move_construction, context["content"], re.IGNORECASE)
        if found:
            utterance = found.group(1).lower()
        validation = self.validate_answer(utterance)
        if self.directions_next_node == None:
            return "Game needs to be aborted"
        if self.current_node == None:
            return "Game needs to be aborted"
        current_location = self.current_node
        if self.ambiguity != None:
            current_location = self.current_node.split("_")[
                0]  ##because if there is ambiguity, the node is saved as "Kitchen_(1,2)"
        if validation != "not valid":
            positive_answer = self.positive_answer
            positive_answer = positive_answer.replace("$DIRECTIONS$", self.directions_next_node)
            if self.graph_type == "named_graph":
                positive_answer = positive_answer.replace("$ANOTHER_ROOM$", current_location)
            utterance = positive_answer
        else:
            negative_answer = self.negative_answer
            negative_answer = negative_answer.replace("$DIRECTIONS$", self.directions_next_node)
            if self.graph_type == "named_graph":
                negative_answer = negative_answer.replace("$SAME_ROOM$", current_location)
            utterance = negative_answer
        return utterance


@dataclass
class TextmapworldSpecificroomGameState(GameState):
    graph_type: str
    initial_position: str
    playerA_initial_prompt: str
    directions: List[Tuple[str, List[str]]]
    ambiguity: Any
    move_construction: str
    stop_construction: str
    reprompting_parameter: bool
    loop_reprompting: str
    maxturns_parameter: bool
    max_turns_reprompting: str
    initial_directions: List[str]
    changed_initial_directions: str
    nodes: List[str]
    visited_nodes: List[str]
    specific_room: str
    steps_made: int = 0
    max_turns: int = 20
    game_error: Optional[List[Dict[str, str | int]]] = None
    game_stop: bool = False
    invalid_response: bool = False
    limit_reached: bool = False

    def __post_init__(self):
        super().__init__()


class textmapworld_specificroom(DialogueGameMaster):
    """
    This class implements a graph traversal game in which player A (DecisionMaker).
    """

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        graph_type = game_instance['Game_Type']
        playerA_initial_prompt = game_instance["Prompt"]
        initial_position = game_instance["Current_Position"]
        directions = ast.literal_eval(game_instance['Directions'])
        ambiguity = game_instance["Ambiguity"]

        if "$INITIAL_ROOM$" in playerA_initial_prompt:
            initial_directions = initial_position
            if ambiguity != None:
                initial_directions = initial_position.split("_")[0]
            playerA_initial_prompt = playerA_initial_prompt.replace("$INITIAL_ROOM$", initial_directions)
        initial_directions = get_directions(initial_position, directions, initial_position)
        changed_initial_directions = string_available_directions(initial_directions)
        playerA_initial_prompt = playerA_initial_prompt.replace("$INITIAL_DIRECTIONS$", changed_initial_directions)
        playerA_initial_prompt = playerA_initial_prompt.replace("$GOAL$", game_instance["Specific_Room"])

        self.state = TextmapworldSpecificroomGameState(
            graph_type=graph_type,
            initial_position=initial_position,
            playerA_initial_prompt=playerA_initial_prompt,
            directions=directions,
            ambiguity=ambiguity,
            move_construction=game_instance["Move_Construction"],
            stop_construction=game_instance["Stop_Construction"],
            reprompting_parameter=game_instance["Loop_Reminder"],
            loop_reprompting=game_instance["Loop_Reminder_Text"],
            maxturns_parameter=game_instance["Max_Turns_Reminder"],
            max_turns_reprompting=game_instance["Max_Turns_Reminder_Text"],
            initial_directions=initial_directions,
            changed_initial_directions=changed_initial_directions,
            nodes=ast.literal_eval(game_instance['Graph_Nodes']),
            visited_nodes=[initial_position],
            specific_room=game_instance["Specific_Room"],
        )

        self.guesser = PathGuesser(self.player_models[0])
        self.describer = PathDescriber(game_instance)
        self.add_player(self.guesser)
        self.add_player(self.describer)

    def _on_before_game(self):
        self.set_context_for(self.guesser, self.state.playerA_initial_prompt)

    def _parse_response(self, player: Player, utterance: str) -> str:
        if player == self.guesser:
            found = None
            utterance = utterance.replace("\n", "").strip()
            if re.search(self.state.stop_construction, utterance, re.IGNORECASE):
                found = re.search(self.state.move_construction, utterance, re.IGNORECASE)
            elif re.search(self.state.move_construction, utterance, re.IGNORECASE):
                found = re.search(self.state.stop_construction, utterance, re.IGNORECASE)
            if found:
                utterance = found.group()
            self.log_to_self("parse", utterance)
        return utterance

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        is_valid = True
        if player == self.guesser:
            stop_action = re.search(self.state.stop_construction, utterance, re.IGNORECASE)
            move_action = re.search(self.state.move_construction, utterance, re.IGNORECASE)
            if move_action and stop_action:
                self.state.invalid_response = True
                is_valid = False
            else:
                count_go = re.findall(self.state.move_construction, utterance, re.IGNORECASE)
                if len(count_go) > 1:
                    self.state.invalid_response = True
                    is_valid = False
                elif not move_action and not stop_action:
                    self.state.invalid_response = True
                    is_valid = False
        elif player == self.describer:
            if utterance == "Game needs to be aborted":
                if self.state.game_error is not None:
                    error_type = self.state.game_error["type"]
                    if error_type == 0:
                        self.log_to_self("Direction not available", "The desired direction is not in available paths")
                self.state.invalid_response = True
                is_valid = False
        if is_valid:
            self.log_to_self("Valid format", "Continue")
        else:
            self.log_to_self("aborted", "abort game")
            self.state.abort()
        return is_valid

    def _on_valid_player_response(self, player: Player, utterance: str):
        """Add the utterance to other player's history, if necessary.
        To do this use the method add_user_message(other_player,utterance)."""
        if player == self.guesser:
            stop_action = re.search(self.state.stop_construction, utterance, re.IGNORECASE)
            if stop_action:
                self.log_to_self("stop", "The guesser decided to stop the game")
                if self.state.visited_nodes[-1].lower() == self.state.specific_room.lower():
                    self.state.succeed()
                else:
                    self.state.failed()
            else:
                self.set_context_for(self.describer, utterance)
        elif player == self.describer:
            if self.state.reprompting_parameter and loop_identification(self.state.visited_nodes, False):
                self.log_to_self("loop_detected", "Loop detected in the visited nodes list")
                self.state.reprompting_parameter = False
                utterance = self.state.loop_reprompting + "\n" + utterance
            if self.state.maxturns_parameter and self.current_round == self.state.max_turns - 2:
                self.state.maxturns_parameter = False
                utterance = self.state.max_turns_reprompting + "\n" + utterance
            self.set_context_for(self.guesser, utterance)

    def _on_after_round(self):
        if self.current_round + 1 >= self.state.max_turns:
            self.log_to_self("turns_limit", str(self.state.max_turns))
            self.state.abort()
        else:
            turn_dict = self.describer.turn_information()
            old_node = turn_dict["from"]
            new_node = turn_dict["to"]
            if old_node != new_node:
                if not self.state.game_stop and not self.state.invalid_response and not self.state.limit_reached and not self.state.game_error:
                    self.log_to_self(type_="move", value=json.dumps({"old": old_node, "new": new_node}))
                    self.state.visited_nodes.append(new_node)
                    if self.state.reprompting_parameter and loop_identification(self.state.visited_nodes, False):
                        self.state.visited_nodes.clear()
                        self.state.reprompting_parameter = True


class GraphGameScorer(GameScorer):

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.nodes = ast.literal_eval(game_instance['Graph_Nodes'])
        self.game_type = game_instance['Game_Type']
        self.ambiguity = game_instance['Ambiguity']
        old_edges = ast.literal_eval(game_instance['Graph_Edges'])
        new_edges = [(edge[1], edge[0]) for edge in old_edges]
        new_edges.extend(old_edges)
        self.edges = new_edges
        self.start = game_instance["Current_Position"]
        self.specifc_room = game_instance['Specific_Room']

    def visited_all(self, visited, to_visit):
        return all([n in visited for n in to_visit])

    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]

    def adj(self, node):
        return set([ed[1] for ed in self.edges if ed[0] == node])

    def find_best_moves(self, current, visited):

        to_visit = [ed[1] for ed in self.edges if ed[0] in visited and ed[1] not in visited]
        start = [current]
        q = Queue()
        q.put(start)
        found = set()
        max_len = 100
        while True:
            n = q.get()
            if len(n) > max_len:
                break
            if self.visited_all(n, to_visit):
                if len(n) > 1:
                    found.add((n[0], n[1]))
                    max_len = len(n)

            avail = self.get_available_moves(n[-1])
            if all([move[1] in n for move in avail]):
                for move in avail:
                    new = deepcopy(n)
                    new.append(move[1])
                    q.put(new)
            else:
                for move in avail:
                    if not move[1] in n:
                        new = deepcopy(n)
                        new.append(move[1])
                        q.put(new)
        return found

    def compute_scores(self, episode_interactions) -> None:

        current = self.start
        seen = {self.start}
        seen.update(self.adj(self.start))
        visited = {self.start}
        visited_list = [self.start]
        valid_moves = 0
        invalid_moves = 0
        stopped = False
        aborted = False
        turns_limit_reached = False
        good_move = []
        loops = []
        count_loops = 0
        success = 0
        loops.append(self.start)
        for turn in episode_interactions["turns"]:
            for event in turn:
                action = event["action"]
                if action["type"] == "aborted":
                    if action["content"]:
                        aborted = True
                if action['type'] == "move":
                    cont = json.loads(action['content'])
                    if not cont["old"] == cont["new"]:
                        valid_moves += 1
                    else:
                        invalid_moves += 1
                    best_moves = self.find_best_moves(current, visited)
                    visited_list.append(cont["new"])
                    new_move = cont["new"]
                    if (current, new_move) in best_moves:
                        if len(visited) != len(self.nodes):
                            good_move.append(True)
                        else:
                            good_move.append(False)
                    else:
                        good_move.append(False)
                    current = cont["new"]
                    seen.update(self.adj(current))
                    loops.append(current)
                    visited.add(current)
                    if loop_identification(loops):
                        count_loops += 1
                        loops.clear()

                if action['type'] == "stop":
                    if action["content"]:
                        stopped = True

                if action['type'] == "turns_limit":
                    turns_limit_reached = True

        for i, val in enumerate(good_move):
            self.log_turn_score(i, "efficient_move", int(val))
        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
        else:
            if not stopped:
                self.log_episode_score(METRIC_ABORTED, 1)
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 0)
            else:
                if self.specifc_room.lower() == visited_list[-1].lower():
                    success = 100
                    self.log_episode_score(METRIC_SUCCESS, 1)
                    self.log_episode_score(METRIC_ABORTED, 0)
                    self.log_episode_score(METRIC_LOSE, 0)
                else:
                    self.log_episode_score(METRIC_SUCCESS, 0)
                    self.log_episode_score(METRIC_ABORTED, 0)
                    self.log_episode_score(METRIC_LOSE, 1)

        exploration = (len(visited) / len(self.nodes) * 100) if len(self.nodes) else 0
        efficiency = (sum(good_move) / len(good_move) * 100) if good_move else 0
        bench_score = (2 * efficiency * exploration / (efficiency + exploration)) if (efficiency + exploration) else 0
        self.log_episode_score('moves', valid_moves + invalid_moves if stopped else np.nan)
        self.log_episode_score('valid_moves', valid_moves if stopped else np.nan)
        self.log_episode_score('invalid_moves', invalid_moves if stopped else np.nan)
        self.log_episode_score('stopped', int(stopped) if stopped else np.nan)
        self.log_episode_score('turns_limit', int(turns_limit_reached) if stopped else np.nan)
        self.log_episode_score('loops', count_loops if stopped else np.nan)
        self.log_episode_score('number_visited', len(visited) if stopped else np.nan)
        self.log_episode_score('seen', len(seen) if stopped else np.nan)
        self.log_episode_score('efficiency', efficiency if stopped else np.nan)
        self.log_episode_score('exploration', exploration if stopped else np.nan)
        self.log_episode_score('old_benchscore', bench_score if stopped else np.nan)
        self.log_episode_score(BENCH_SCORE, success if stopped else np.nan)


class GraphGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return textmapworld_specificroom(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return GraphGameScorer(self.game_name, experiment, game_instance)


def main():
    # select one experiment and instance
    experiments = file_utils.load_json("in/instances_specificroom.json", "textmapworld")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"]
    master = textmapworld_specificroom(experiment_1, ["mock", "mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()
