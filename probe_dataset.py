from statistics import stdev
from numpy import mean
import matplotlib.pyplot as plt
import jsonlines
from download_dataset import JSONL_FILE
from collections import Counter

num_edits = []

# {"files": [{"id": "edit_turn1", "url": "images/multi-turn/101_attemptB_turn1.png"}, {"id": "edit_turn2", "url": "images/multi-turn/101_attemptB_turn2.png"}, {"id": "edit_turn3", "url": "images/multi-turn/101_attemptB_turn3.png"}, {"id": "edit_turn4", "url": "images/multi-turn/101_attemptB_turn4.png"}, {"id": "final_image", "url": "images/multi-turn/101_attemptB_turn5.png"}, {"id": "original_input_image", "url": "https://farm8.staticflickr.com/2915/14573719235_6cfb811e3c_o.jpg"}], "metadata_edit_turn_prompts": ["Shift the entire image's color tone to a much warmer, golden hue", "Zoom in closely on the front and side of the main silver race car", "Outpaint the canvas to show more of the outdoor event and spectators behind the cars", "Transform the entire scene into a detailed comic book art style", "Replace all the classic race cars with modern, futuristic electric race cars"]}


with jsonlines.open(JSONL_FILE, "r") as f:
    for _, item in enumerate(f):
        num_edits.append(len(item["metadata_edit_turn_prompts"]))

counts = Counter(num_edits)
item_mean = mean(num_edits)
item_sd = stdev(num_edits)
print(counts)

# Create a bar plot
plt.bar(counts.keys(), counts.values())
plt.xlabel("Edit turns")
plt.ylabel("Count")
plt.title(f"Distribution of Edit turns (mean={item_mean:.4f}, sd={item_sd:.4f})")
plt.savefig("distribution.png")

