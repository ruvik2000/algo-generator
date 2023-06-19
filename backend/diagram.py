from graphviz import Digraph

# Sample sequence representation
sequence = "<start> <process: Initialize count to 0> <decision: Check if count is less than 10> <true> <process: Print 'Hello World'> <process: Increment count> <loop> <false> <end>"

# Parse the sequence
tokens = sequence.split()

# Create a directed graph for the flowchart
flowchart = Digraph()

# Iterate through the tokens and build the flowchart
prev_token = None
for token in tokens:
    # Extract symbol type and description
    symbol_type, description = token[1:-1].split(":", 1) if ":" in token else (token[1:-1], "")

    # Add symbol to the flowchart
    flowchart.node(token, label=description)

    # Add connector if applicable
    if prev_token and not (prev_token == "<true>" or prev_token == "<false>"):
        flowchart.edge(prev_token, token)

    prev_token = token

# Render the flowchart as an SVG
flowchart.render('flowchart.svg', view=True)
