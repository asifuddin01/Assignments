
#Task 1


import random
import math



# Chip Components
blocks = {
    'ALU': (5, 5),
    'Cache': (7, 4),
    'Control Unit': (4, 4),
    'Register File': (6, 6),
    'Decoder': (5, 3),
    'Floating Unit': (5, 5)
}



block_names = list(blocks.keys())

# blocks connection
connections = [
    ('Register File', 'ALU'),
    ('Control Unit', 'ALU'),
    ('ALU', 'Cache'),
    ('Register File', 'Floating Unit'),
    ('Cache', 'Decoder'),
    ('Decoder', 'Floating Unit')
]



gridSize = 25
popSize = 6
maxGen = 15

# Random Layout
def random_layout():
    layout = []

    for name in block_names:

        width, height = blocks[name]

        x = random.randint(0, gridSize - width)
        y = random.randint(0, gridSize - height)
        layout.append((x, y))

    return layout

# Calculate Score

def get_score(layout):

    positions = {}

    for i in range(len(block_names)):
        name = block_names[i]

        x, y = layout[i]
        w, h = blocks[name]
        positions[name] = (x, y, w, h)

    # Check overlaps
    overlaps = 0
    for i in range(len(block_names)):
        for j in range(i+1, len(block_names)):

            x1, y1, w1, h1 = positions[block_names[i]]
            x2, y2, w2, h2 = positions[block_names[j]]


            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                overlaps += 1

    # Calculate wire lengths

    wire_length = 0
    for block1, block2 in connections:
        x1, y1, w1, h1 = positions[block1]
        x2, y2, w2, h2 = positions[block2]

        # Get center points

        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)

        # Add distance between centers
        wire_length += math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

    # Calculate area
    all_x = [pos[0] for pos in layout]
    all_y = [pos[1] for pos in layout]

    max_x = max([all_x[i] + blocks[block_names[i]][0] for i in range(len(block_names))])
    max_y = max([all_y[i] + blocks[block_names[i]][1] for i in range(len(block_names))])
    area = (max_x - min(all_x)) * (max_y - min(all_y))

    # Final score (target: minimize these)
    score = -(1000 * overlaps + 2 * wire_length + 1 * area)
    return score, overlaps, wire_length, area

#GENETIC
def combine_layouts(layout1, layout2):
    point = random.randint(1, len(layout1)-1)
    child1 = layout1[:point] + layout2[point:]
    child2 = layout2[:point] + layout1[point:]
    return child1, child2

def change_layout(layout):

    if random.random() < 0.3:  # 30% chance to change
        i = random.randint(0, len(layout)-1)
        name = block_names[i]

        w, h = blocks[name]
        layout[i] = (random.randint(0, gridSize-w), random.randint(0, gridSize-h))


    return layout


population = [random_layout() for _ in range(popSize)]

for generation in range(maxGen):

    scored = []
    for layout in population:
        score, ov, wire, area = get_score(layout)
        scored.append((layout, score, ov, wire, area))

    # Sort by score (best first)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top 2
    new_pop = [scored[0][0], scored[1][0]]

    # Make random children
    while len(new_pop) < popSize:

        parent1 = random.choice(population)
        parent2 = random.choice(population)

        child1, child2 = combine_layouts(parent1, parent2)
        new_pop.append(change_layout(child1))

        if len(new_pop) < popSize:
            new_pop.append(change_layout(child2))

    population = new_pop

# Best result
best = max([(layout, *get_score(layout)) for layout in population], key=lambda x: x[1])

print("=== BEST LAYOUT ===")
print(f"Score: {best[1]}")
print(f"Overlaps: {best[2]}")
print(f"Wire Length: {round(best[3], 2)}")
print(f"Area: {best[4]}\n")

print("Block Positions:")
for i in range(len(block_names)):
    print(f"{block_names[i]:15} at ({best[0][i][0]:2}, {best[0][i][1]:2})")

#Task 2

import random
import math



# Chip Components
blocks = {
    'ALU': (5, 5),
    'Cache': (7, 4),
    'Control Unit': (4, 4),
    'Register File': (6, 6),
    'Decoder': (5, 3),
    'Floating Unit': (5, 5)
}



block_names = list(blocks.keys())

# blocks connection
connections = [
    ('Register File', 'ALU'),
    ('Control Unit', 'ALU'),
    ('ALU', 'Cache'),
    ('Register File', 'Floating Unit'),
    ('Cache', 'Decoder'),
    ('Decoder', 'Floating Unit')
]



gridSize = 25
popSize = 6
maxGen = 15

# Random Layout
def random_layout():
    layout = []

    for name in block_names:

        width, height = blocks[name]

        x = random.randint(0, gridSize - width)
        y = random.randint(0, gridSize - height)
        layout.append((x, y))

    return layout

# Calculate Score

def get_score(layout):

    positions = {}

    for i in range(len(block_names)):
        name = block_names[i]

        x, y = layout[i]
        w, h = blocks[name]
        positions[name] = (x, y, w, h)

    # Check overlaps
    overlaps = 0
    for i in range(len(block_names)):
        for j in range(i+1, len(block_names)):

            x1, y1, w1, h1 = positions[block_names[i]]
            x2, y2, w2, h2 = positions[block_names[j]]


            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                overlaps += 1

    # Calculate wire lengths

    wire_length = 0
    for block1, block2 in connections:
        x1, y1, w1, h1 = positions[block1]
        x2, y2, w2, h2 = positions[block2]

        # Get center points

        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)

        # Add distance between centers
        wire_length += math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

    # Calculate area
    all_x = [pos[0] for pos in layout]
    all_y = [pos[1] for pos in layout]

    max_x = max([all_x[i] + blocks[block_names[i]][0] for i in range(len(block_names))])
    max_y = max([all_y[i] + blocks[block_names[i]][1] for i in range(len(block_names))])
    area = (max_x - min(all_x)) * (max_y - min(all_y))

    # Final score (target: minimize these)
    score = -(1000 * overlaps + 2 * wire_length + 1 * area)
    return score, overlaps, wire_length, area

#GENETIC
def combine_layouts(layout1, layout2):

    i, j = sorted(random.sample(range(len(layout1)), 2))

    child1 = layout1[:i] + layout2[i:j] + layout1[j:]
    child2 = layout2[:i] + layout1[i:j] + layout2[j:]

    return child1, child2


def change_layout(layout):

    if random.random() < 0.3:  # 30% chance to change
        i = random.randint(0, len(layout)-1)
        name = block_names[i]

        w, h = blocks[name]
        layout[i] = (random.randint(0, gridSize-w), random.randint(0, gridSize-h))


    return layout


population = [random_layout() for _ in range(popSize)]

for generation in range(maxGen):

    scored = []
    for layout in population:
        score, ov, wire, area = get_score(layout)
        scored.append((layout, score, ov, wire, area))

    # Sort by score (best first)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top 2
    new_pop = [scored[0][0], scored[1][0]]

    # Make random children
    while len(new_pop) < popSize:

        parent1 = random.choice(population)
        parent2 = random.choice(population)

        child1, child2 = combine_layouts(parent1, parent2)
        new_pop.append(change_layout(child1))

        if len(new_pop) < popSize:
            new_pop.append(change_layout(child2))

    population = new_pop

# Best result
best = max([(layout, *get_score(layout)) for layout in population], key=lambda x: x[1])

print("=== BEST LAYOUT ===")
print(f"Score: {best[1]}")
print(f"Overlaps: {best[2]}")
print(f"Wire Length: {round(best[3], 2)}")
print(f"Area: {best[4]}\n")

print("Block Positions:")
for i in range(len(block_names)):
    print(f"{block_names[i]:15} at ({best[0][i][0]:2}, {best[0][i][1]:2})")

