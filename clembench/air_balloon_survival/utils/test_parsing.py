import re
# ---------------------------
# Simulated game settings (like self.game_instance)
game_instance = {
    "strategic_reasoning": True,
    "require_argument": True,
    "parse_error_prompt": "Parsing failed!"
}

strategic_reasoning_tag = r"^\s*STRATEGIC REASONING:\s*\{(?:[^{}]|\{[^{}]*\})*\}"
argument_tag = r"ARGUMENT:\s*\{(?:[^{}]|\{[^{}]*\})*\}"
proposal_tag = r"PROPOSAL:\s*\{.*?\}"

# ---------------------------
# Function that mimics your original parsing logicÖ
def parse_response(response):
    tags_dict = {}

    if game_instance["strategic_reasoning"]:
        tags_dict[strategic_reasoning_tag] = []
        if not re.match(strategic_reasoning_tag, response, re.DOTALL):
            print("🚨 Parsing failed: STRATEGIC REASONING not at start.")
            return False

    if game_instance["require_argument"]:
        if not re.search(argument_tag, response, re.DOTALL):
            print("🚨 Parsing failed: ARGUMENT not found.")
            return False
        tags_dict[argument_tag] = []

    # Always expect a proposal
    tags_dict[proposal_tag] = []

    remaining_text = response
    for tag_pattern in tags_dict.keys():
        remaining_text = re.sub(tag_pattern, "", remaining_text, flags=re.DOTALL)

    if remaining_text.strip():
        print("🚨 Parsing failed: leftover content after stripping.")
        return False

    print("✅ Parsing success: all tags found and no leftover content.")
    return True

# ---------------------------
# Test inputs
valid_response = """
STRATEGIC REASONING: {
- I rejected their proposal because:
  - It included costly items
  - It ignored preferences
- I built a set, {'villa', 'lantern'}
}
ARGUMENT: {
This shows efficiency with {'villa', 'lantern'}
}


PROPOSAL: {'villa', 'lantern'}
"""

fail_response_extra_text = """
STRATEGIC REASONING: {
- Decision based on cost
}
This is some stray text.
ARGUMENT: {
Efficiency argument.
}
PROPOSAL: {'villa', 'lantern'}
"""

fail_response_no_argument = """
STRATEGIC REASONING: {
Some strategy.
}
PROPOSAL: {'villa', 'lantern'}
"""

fail_response_non_tagged_middle = """
STRATEGIC REASONING: {
Initial analysis done.
}
Some completely untagged text right here.
ARGUMENT: {
Efficiency shown.
}
PROPOSAL: {'villa', 'lantern'}
"""

fail_response_sr_not_at_start = """
ARGUMENT: {
We have an efficiency argument first.
}
STRATEGIC REASONING: {
Now here comes the strategic reasoning afterwards.
}
PROPOSAL: {'villa', 'lantern'}
"""

# ---------------------------
# Run tests
print("\n--- Valid case ---")
parse_response(valid_response)

print("\n--- Fail case: extra text ---")
parse_response(fail_response_extra_text)

print("\n--- Fail case: missing ARGUMENT ---")
parse_response(fail_response_no_argument)

print("\n--- Fail case: non-tagged sequence in the middle ---")
parse_response(fail_response_non_tagged_middle)

print("\n--- Fail case: STRATEGIC REASONING not at start ---")
parse_response(fail_response_sr_not_at_start)
