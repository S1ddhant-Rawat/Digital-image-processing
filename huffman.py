import heapq, pygame
from collections import Counter

TEXT = "ABRACADABRA"                                     # change this to encode your own text

class Node:
    def __init__(self, freq, char=None, left=None, right=None, step=0):
        self.freq, self.char, self.left, self.right, self.step = freq, char, left, right, step

# --- Huffman algorithm: repeatedly merge the two smallest nodes ---
heap = [(f, i, Node(f, ch)) for i, (ch, f) in enumerate(sorted(Counter(TEXT).items()))]
heapq.heapify(heap)
count, merges = len(heap), []
while len(heap) > 1:
    f1, _, a = heapq.heappop(heap)                      # smallest
    f2, _, b = heapq.heappop(heap)                      # second smallest
    merged = Node(f1 + f2, None, a, b, step=len(merges) + 1)
    merges.append((a, b, merged))
    heapq.heappush(heap, (merged.freq, count, merged))
    count += 1
root = heap[0][2]

codes = {}
def assign(n, code=""):                                 # left edge = 0, right edge = 1
    if n.char is not None: codes[n.char] = code or "0"
    else: assign(n.left, code + "0"); assign(n.right, code + "1")
assign(root)

nodes, leaf_count = [], [0]
GAP = 800 / max(1, len(codes) - 1)
def layout(n, depth):                                   # position nodes for drawing
    n.y = 150 + depth * 85
    if n.char is not None: n.x = 50 + leaf_count[0] * GAP; leaf_count[0] += 1
    else: layout(n.left, depth + 1); layout(n.right, depth + 1); n.x = (n.left.x + n.right.x) / 2
    nodes.append(n)
layout(root, 0)

encoded = "".join(codes[c] for c in TEXT)
pygame.init()
screen = pygame.display.set_mode((900, 640))
font, small = pygame.font.SysFont("arial", 20), pygame.font.SysFont("arial", 18, bold=True)
clock, frame = pygame.time.Clock(), 0
pygame.display.set_caption("Huffman coding")

while not pygame.event.peek(pygame.QUIT):
    pygame.event.pump()
    s = frame // 70                                     # one merge every 70 frames
    done = s > len(merges)
    screen.fill((250, 249, 245))

    if s == 0: msg = f'Text: "{TEXT}"   Start: every symbol is a node with its frequency'
    elif not done: msg = f"Step {s}: merge the two smallest nodes ({merges[s-1][0].freq} + {merges[s-1][1].freq} = {merges[s-1][2].freq})"
    else: msg = "Done: left edge = 0, right edge = 1. Read the path from the top to get each code."
    screen.blit(font.render(msg, True, (31, 35, 40)), (15, 12))

    for n in nodes:                                     # edges of merged nodes
        if n.char is None and n.step <= s:
            for child, bit in ((n.left, "0"), (n.right, "1")):
                color = (242, 177, 52) if n.step == s else (130, 130, 130)
                pygame.draw.line(screen, color, (n.x, n.y), (child.x, child.y), 4 if n.step == s else 2)
                if done:
                    screen.blit(small.render(bit, True, (200, 60, 60)), ((n.x + child.x) / 2 - 12, (n.y + child.y) / 2 - 10))
    for n in nodes:                                     # nodes
        if n.step > s: continue
        fill = (242, 177, 52) if n.step == s and n.step > 0 else (47, 143, 107) if n.char else (70, 110, 200)
        pygame.draw.circle(screen, fill, (n.x, n.y), 26)
        label = f"{n.char}:{n.freq}" if n.char else str(n.freq)
        screen.blit(small.render(label, True, (255, 255, 255)), small.render(label, True, (0, 0, 0)).get_rect(center=(n.x, n.y)))
        if done and n.char:
            screen.blit(font.render(codes[n.char], True, (47, 143, 107)), (n.x - 20, n.y + 32))
    if done:
        screen.blit(font.render(f"Encoded: {encoded}", True, (31, 35, 40)), (15, 560))
        screen.blit(font.render(f"{len(encoded)} bits with Huffman vs {8 * len(TEXT)} bits in plain 8-bit text", True, (107, 114, 128)), (15, 592))

    pygame.display.flip()
    frame += 1
    if s > len(merges) + 4: frame = 0                   # replay
    clock.tick(60)
pygame.quit()