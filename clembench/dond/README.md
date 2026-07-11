A Negotiation Game: Deal or No Deal (DoND)
==========================================

Implemented by: Roland Bernard

In this game, two players are given a small set of items. Each player is also given a different private value function, mapping each item type to a non-negative integer indicating how much the player values it. The players are then able to exchange messages, which the goal of agreeing on how to split the items between them. At any point during this conversation, or when requested to after a time limit, the players can make a secret proposal indicating which items they would like to receive. Once one of the players has made the proposal, the other one is asked to make one as well. The player is not given the proposal of the other player, meaning the prior discussion must have made it clear how they should divide the items.

Once both players have made their proposal, the game is finished. The game master will then check whether the two proposals are complementary, i.e., there are sufficient items to satisfy both proposals. If they are compatible, then the players each receive points based on their private value functions and the items they received. If the proposals are conflicting, then both players will receive zero points. There are three different versions of this game implemented. From the perspective of the players, the only difference is that they are asked to optimize for a different metric in the initial prompt explaining the rules.

* **Semi-competitive**: In this game mode, the players are asked to maximize their own score. This is the more standard scenario, in which not reaching any agreement is clearly worse for both players, but there is still a competitive aspect to the game.
* **Cooperative**: In the cooperative setting, the players are asked to optimize for the sum of their score and the other players score. This means the game is purely cooperative, and both players should want the same optimal outcome.
* **Competitive**: In this scenario the game is made strictly competitive by asking the players to maximize the difference between their score and the opponents score. This results in a zero-sum game, where non-agreement may sometimes even be the optimal strategy.

The game tests a LLM's ability to describe their preferences, understand the preferences of the other player, and reach unanimous agreements on how to split the items. It also tests whether the models are able to stay grounded. For example, if the players invent new items or misrepresent their value functions, this would negatively impact their performance in the game. This game also challenges the LLM's ability to follow rules and stick to agreements made with the other players.

## Instantiation

The game instances are created with a fixed number of 5 turns, i.e., each player is allowed to send 5 messages before the game master will ask them to make their proposal. Each game instance uses between 3 and 5 unique item types, samples from a set of 100 possible item words (found in `resources/en/possible_items.json`). In total, each game contains between 5 and 8 items. The value functions for the two players are randomly generated subject to the following constraints:

* *Each player can obtain a maximum score of exactly 10.* This happens when they are given all the available items.
* *Every item has non-zero value to at least one player.* This ensures that all items are worth discussing over.
* *There is at least one item that has a non-zero value to both players.* This condition ensures that only one of the two players can achieve the maximum possible score of 10. This makes the negotiations competitive, ensuring that there is no trivial agreement.

Game instances can be generated for each of the tree game modes, and for any of three possible languages, English, German, and Italian. Other than the initial prompt there is no difference in the way instances are generated for the different game modes, and there is no difference between the instances generated for different languages.

## Evaluation

The performance if the LLMs is measured using the following episode-level metrics:

* **Aborted**: Whether the game has been terminated prematurely due to rule violations of one of the players.
* **Lose**: Whether the game has completed with both players submitting proposals, but the proposals where conflicting.
* **Success**: Whether the game finished and a valid agreement was reached.
* **Pareto Optimal**: Whether the game resulted in a Pareto optimal outcome, i.e., no player's score can be improved further without decreasing another player's score.
* **Main Score**: The main quality score for this game is computed based on the maximum Pareto improvement as $100 - 100 \cdot \frac{\text{Maximum Pareto Improvement}}{\text{Maximum Score per Player}}$. The maximum Pareto improvement is the maximum improvement that can be achieved to one of the player's scores without decreasing the score of the other player. As such, this quality score reflects how close the achieved agreement is to the optimal outcome, with higher scores indicating better performance relative to the maximum possible improvement.

