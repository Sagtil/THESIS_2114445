"""
RAID Dataset Generator — Referential Availability in Implied Discourse
Ties Bogaers (2114445), Tilburg University · 2026

Split design:
  - explicit_leave & baseline: lexically split into train/test (probe poles)
  - implied_leave, implied_cancel, disengaged: generalisation only (no split)
"""

import csv
import random
from itertools import product
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# 1. NAMES — NO OVERLAP between train and test
# ═══════════════════════════════════════════════════════════════════

MALE_NAMES = {
    "train": [
        "Aaron", "Adam", "Alex", "Ben", "Brandon", "Caleb", "Cole", "Connor",
        "Dan", "Dylan", "Edward", "Ethan", "Finn", "Gavin", "Henry", "Hugo",
        "Isaac", "Jack", "James", "Jason", "Marcus", "Kevin", "Kyle", "Logan",
        "Lucas", "Luke", "Mark", "Matthew", "Nathan", "Noah", "Nolan", "Oscar",
        "Owen", "Patrick", "Paul", "Robert", "Ryan", "Sam", "Scott", "Sean",
        "Tom", "Thomas", "Tyler", "Victor", "Rick", "Will", "Xavier", "Daniel",
        "Zach", "Zane",
    ],
    "test": [
        "Arthur", "Bill", "Charlie", "David", "Evan", "Frank", "George",
        "Harry", "Ian", "John", "Martin", "Michael", "Peter", "Philip",
        "Sebastian", "Richard", "Steven", "Timothy", "Bruce", "Walter",
    ]
}

FEMALE_NAMES = [
    "Anna", "Claire", "Emma", "Hannah", "Julia", "Lisa", "Louise",
    "Mary", "Nina", "Olivia", "Rachel", "Sarah", "Sophie",
]

# ═══════════════════════════════════════════════════════════════════
# 2. LOCATIONS (26 total → 18 train / 8 test)
# ═══════════════════════════════════════════════════════════════════

LOCATIONS = {
    "train": [
        "cafe", "classroom", "hallway", "kitchen", "library", "office",
        "lobby", "studio", "pub", "bar", "lecture hall", "shop",
        "living room", "apartment", "restaurant", "house",
        "football stadium", "museum",
    ],
    "test": [
        "club", "grocery store", "bedroom", "balcony", "teahouse",
        "church", "temple", "swimming pool",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# 3. TOPICS (29 total → 20 train / 9 test)
# ═══════════════════════════════════════════════════════════════════

TOPICS = {
    "train": [
        "a funny story", "some news", "the plan", "the project",
        "a secret", "the schedule", "the problem", "a fight",
        "a hobby", "the education system", "a sport", "the movie",
        "social media", "some music", "football", "a band",
        "the local artists", "the politics", "the festivals", "gossip",
    ],
    "test": [
        "a car", "a bike", "the motorway", "a necklace",
        "the hairstyles", "the latest fashion", "a beer",
        "web browsers", "the weather",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# 4. STAYER PHRASES (43 total → 30 train / 13 test)
#    Used for the staying character in ALL conditions (including baseline)
# ═══════════════════════════════════════════════════════════════════

STAYER_PHRASES = {
    "train": [
        "laughed and continued the discussion",
        "smiled and kept the conversation going",
        "kept eye contact and added another point",
        "leaned forward and listened attentively",
        "nodded along and kept engaging with the topic",
        "asked a follow-up and stayed engaged",
        "responded with interest and kept chatting",
        "reacted enthusiastically and stayed involved",
        "chuckled and pressed for more details",
        "grinned and offered his own perspective",
        "tilted his head and kept the momentum going",
        "raised an eyebrow and asked another question",
        "took a sip of his drink and deepened the dialogue",
        "gestured animatedly and steered the chat forward",
        "sat back comfortably and kept the banter flowing",
        "paused thoughtfully and brought up a related story",
        "smiled warmly and stayed fully present",
        "laughed out loud and continued bouncing ideas around",
        "hummed in agreement and anchored the conversation",
        "kept a relaxed posture and didn't break focus",
        "leaned against the table and maintained the easy flow",
        "gave a soft smile and kept the energy up",
        "leaned in closer and kept probing the topic",
        "mirrored his excitement and fueled the discussion",
        "let out a sigh of relief and kept talking openly",
        "smiled encouragingly and kept the energy going",
        "gave a quick nod and drove the point home",
        "looked thoughtful and kept exploring the idea",
        "shifted closer and maintained the lively exchange",
        "beamed brightly and stayed completely captivated",
    ],
    "test": [
        "playfully rolled his eyes and kept the joke going",
        "adjusted his posture and held his ground in the debate",
        "let out a soft chuckle and held the conversational space",
        "relaxed his shoulders and stayed locked in the moment",
        "perked up and continued the rapid-fire banter",
        "tilted back his chair and stayed actively involved",
        "took a quick breath and launched into the next point",
        "offered a knowing look and kept the dialogue alive",
        "laughed lightly and stuck around for the rest of the story",
        "nodded vigorously and kept validating the point",
        "clapped his hands together and kept the enthusiasm high",
        "gave a reassuring smile and stayed right there in the moment",
        "chuckled knowingly and didn't let the silence settle",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# 5. EXPLICIT LEAVE (23 total → 16 train / 7 test)
# ═══════════════════════════════════════════════════════════════════

EXPLICIT_LEAVE = {
    "train": [
        "checked his phone, said he was late, and left",
        "said he had to go and started gathering his things",
        "stood up, apologized, and headed out",
        "stood up and stepped away politely",
        "put on his jacket and walked toward the door",
        "said goodbye and stepped away",
        "grabbed his things and walked right out",
        "checked his watch and hurried for the exit",
        "gathered his belongings and took off",
        "said a final goodbye and headed out",
        "packed up his stuff and made his departure",
        "turned on his heel and walked out the door",
        "gave a quick nod and made his exit",
        "wrapped things up and walked away entirely",
        "slung his bag over his shoulder and departed",
        "let out a sigh and headed straight out",
    ],
    "test": [
        "grabbed his keys and completely left",
        "stood up, stretched, and took his leave",
        "mumbled a quick excuse and slipped out",
        "signaled he was done and walked right out",
        "waved goodbye and disappeared through the doors",
        "collected his items and swiftly departed",
        "checked the time and rushed out",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# 6. IMPLIED LEAVE (22 total → 15 train / 7 test)
# ═══════════════════════════════════════════════════════════════════

IMPLIED_LEAVE = {
    "train": [
        "glanced at the time and mentioned his first meeting was in ten minutes",
        "checked the time and muttered that he was cutting it close",
        "said he had an early start tomorrow and sighed",
        "mentioned he was supposed to call someone right at the hour",
        "heard his phone buzz and his expression tightened",
        "looked at his phone and muttered that his ride was waiting",
        "stifled a yawn and said he couldn't stay much longer",
        "checked his schedule and noted he had to be somewhere in five minutes",
        "looked out the window and mentioned he needed to beat the traffic",
        "scrolled through his texts and sighed about an urgent errand",
        "glanced at the door and mentioned having to meet a friend",
        "checked the transit app and pointed out his next bus was almost here",
        "sighed heavily and mentioned he had to be up before sunrise",
        "looked at his agenda and grimaced at his next appointment",
        "checked the time and mentioned it was getting pretty late",
    ],
    "test": [
        "said he might miss his last train if things ran long",
        "noted his parking meter was close to expiring and frowned",
        "glanced at his watch and warned that he only had a few minutes left",
        "shifted his weight and brought up the massive workload waiting at the office",
        "checked his notifications and said he was being called away",
        "rubbed the back of his neck and muttered about an upcoming deadline",
        "looked at his screen and apologized that he had to get moving",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# 7. CANCEL SUFFIXES — one per implied_leave phrase
# ═══════════════════════════════════════════════════════════════════

CANCEL_SUFFIXES = {
    # TRAIN
    "glanced at the time and mentioned his first meeting was in ten minutes":
        ", but then said it had been moved",
    "checked the time and muttered that he was cutting it close":
        ", then smiled because he had more time",
    "said he had an early start tomorrow and sighed":
        ", then said he could sleep in after all",
    "mentioned he was supposed to call someone right at the hour":
        ", then said they postponed it",
    "heard his phone buzz and his expression tightened":
        ", then relaxed because it was nothing urgent",
    "looked at his phone and muttered that his ride was waiting":
        ", then laughed because the driver cancelled",
    "stifled a yawn and said he couldn't stay much longer":
        ", then perked up and said he had a second wind",
    "checked his schedule and noted he had to be somewhere in five minutes":
        ", but then realized the event was actually tomorrow",
    "looked out the window and mentioned he needed to beat the traffic":
        ", then checked the map and saw the roads were clear",
    "scrolled through his texts and sighed about an urgent errand":
        ", then shrugged and said it could wait until Monday",
    "glanced at the door and mentioned having to meet a friend":
        ", then got a message saying they were running late",
    "checked the transit app and pointed out his next bus was almost here":
        ", but then decided he would rather just catch the next one",
    "sighed heavily and mentioned he had to be up before sunrise":
        ", but then remembered tomorrow was a day off after all",
    "looked at his agenda and grimaced at his next appointment":
        ", but then saw it had been rescheduled to next week",
    "checked the time and mentioned it was getting pretty late":
        ", then shrugged and said one more round couldn't hurt",
    # TEST
    "said he might miss his last train if things ran long":
        ", then realized it was running later tonight",
    "noted his parking meter was close to expiring and frowned":
        ", then said he had already extended it",
    "glanced at his watch and warned that he only had a few minutes left":
        ", then took his coat back off and sat back down",
    "shifted his weight and brought up the massive workload waiting at the office":
        ", but then said his colleague offered to cover most of it",
    "checked his notifications and said he was being called away":
        ", then realized it was just a group chat and laughed it off",
    "rubbed the back of his neck and muttered about an upcoming deadline":
        ", then smiled because he had already finished the hardest part",
    "looked at his screen and apologized that he had to get moving":
        ", then realized the notification was just spam and relaxed",
}

IMPLIED_CANCEL = {
    split: [base + CANCEL_SUFFIXES[base] for base in phrases]
    for split, phrases in IMPLIED_LEAVE.items()
}

# ═══════════════════════════════════════════════════════════════════
# 8. DISENGAGED (19 total → 13 train / 6 test)
# ═══════════════════════════════════════════════════════════════════

DISENGAGED = {
    "train": [
        "started typing intensely and barely looked up",
        "began reading an email closely and stopped responding",
        "kept scrolling through messages and no longer reacted",
        "focused on troubleshooting something on his laptop and stopped engaging",
        "took a call and focused entirely on the other person",
        "put in earbuds and became absorbed in what he was listening to",
        "stared blankly at his screen and zoned out completely",
        "opened a document and shut out the rest of the room",
        "started watching a video and tuned out the conversation",
        "pulled out a book and got completely lost in the pages",
        "opened an app on his phone and completely checked out",
        "started organizing his papers and gave short, distracted nods",
        "slid his headphones over his ears and closed his eyes",
    ],
    "test": [
        "began jotting down notes and ignored the chatter around the table",
        "got sucked into an article and gave nothing but one-word answers",
        "crossed his arms, stared out the window, and stopped listening",
        "focused entirely on his food and let the conversation drop",
        "started texting rapidly and mentally left the conversation",
        "turned his back slightly to focus on his work and stayed silent",
    ],
}

# ═══════════════════════════════════════════════════════════════════
# 9. CONDITIONS
#    baseline uses STAYER_PHRASES (no separate phrase pool)
# ═══════════════════════════════════════════════════════════════════

# Pole conditions: lexically split into train/test for probe training
POLE_CONDITIONS = {
    "explicit_leave": EXPLICIT_LEAVE,
    "baseline":       STAYER_PHRASES,  # both characters use stayer phrases
}

# Generalisation conditions: no train/test split, all used for evaluation
GEN_CONDITIONS = {
    "implied_leave":  IMPLIED_LEAVE,
    "implied_cancel": IMPLIED_CANCEL,
    "disengaged":     DISENGAGED,
}

# ═══════════════════════════════════════════════════════════════════
# 10. STRUCTURAL VARIABLES (shared, not split)
# ═══════════════════════════════════════════════════════════════════

STRUCTURES = ["while", "but", "whereas", "twosent"]
ROLE_ORDERS = ["changer_first", "stayer_first"]
S3_VARIANTS = ["there", "nothere"]

TRANSITIONS = [
    "After a while,",
    "A bit later,",
    "Some time later,",
    "A little while later,",
    "Eventually,",
    "Later on,",
    "Once some time had passed,",
    "After some time,",
]

OPENING_FRAMES = [
    "{speaker} was talking to {a} and {b} in the {location}.",
    "{speaker} was chatting with {a} and {b} in the {location}.",
    "{speaker} was having a conversation with {a} and {b} in the {location}.",
    "{speaker} was in the {location} with {a} and {b}, deep in conversation.",
    "{speaker}, {a}, and {b} were talking in the {location}.",
]

CLOSING_FRAMES = [
    "{speaker} continued talking to him{s3} about {topic}.",
    "{speaker} went on talking to him{s3} about {topic}.",
    "{speaker} kept discussing {topic} with him{s3}.",
    "{speaker} carried on the conversation with him{s3} about {topic}.",
]

QUESTION = " Question: Who is 'him'? Answer:"

# ═══════════════════════════════════════════════════════════════════
# 11. SENTENCE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════

def build_middle(changer_name, changer_phrase, stayer_name, stayer_phrase,
                 structure, role_order):
    if role_order == "changer_first":
        first_name, first_phrase = changer_name, changer_phrase
        second_name, second_phrase = stayer_name, stayer_phrase
    else:
        first_name, first_phrase = stayer_name, stayer_phrase
        second_name, second_phrase = changer_name, changer_phrase

    first_action = f"{first_name} {first_phrase}"
    second_action = f"{second_name} {second_phrase}"

    if structure == "while":
        return f"While {first_action}, {second_action}."
    elif structure == "but":
        return f"{first_action.rstrip('.')}, but {second_action}."
    elif structure == "whereas":
        return f"{first_action.rstrip('.')}, whereas {second_action}."
    elif structure == "twosent":
        fa = first_action if first_action.endswith(".") else first_action + "."
        sa = second_action if second_action.endswith(".") else second_action + "."
        return f"{fa} {sa}"
    else:
        raise ValueError(f"Unknown structure: {structure}")


def build_prompt(speaker, name_a, name_b, location, changer_name,
                 changer_phrase, stayer_name, stayer_phrase, structure,
                 role_order, s3_variant, topic, rng):
    opening = rng.choice(OPENING_FRAMES).format(
        speaker=speaker, a=name_a, b=name_b, location=location
    )
    middle = build_middle(
        changer_name, changer_phrase, stayer_name, stayer_phrase,
        structure, role_order
    )
    transition = rng.choice(TRANSITIONS)
    s3 = " there" if s3_variant == "there" else ""
    closing = rng.choice(CLOSING_FRAMES).format(
        speaker=speaker, s3=s3, topic=topic
    )
    return f"{opening} {middle} {transition} {closing}{QUESTION}"


# ═══════════════════════════════════════════════════════════════════
# 12. GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════

def get_antecedent_label(condition, stayer_name, changer_name):
    if condition in ("explicit_leave", "implied_leave", "disengaged"):
        return stayer_name
    elif condition in ("implied_cancel", "baseline"):
        return "AMBIGUOUS"
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════════════
# 13. GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_pole_samples(split, target_per_condition, rng):
    """
    Generate samples for pole conditions (explicit_leave, baseline).
    These use the train/test lexical split for names, locations, etc.
    """
    samples = []
    male_names = MALE_NAMES[split]
    locations = LOCATIONS[split]
    topics = TOPICS[split]
    stayer_phrases = STAYER_PHRASES[split]

    structural_combos = list(product(STRUCTURES, ROLE_ORDERS, S3_VARIANTS))

    for condition_name, condition_phrases in POLE_CONDITIONS.items():
        changer_phrases = condition_phrases[split]

        # For baseline, both characters use stayer phrases
        if condition_name == "baseline":
            stayer_phrases_to_use = STAYER_PHRASES[split]
        else:
            stayer_phrases_to_use = stayer_phrases

        for i in range(target_per_condition):
            speaker = rng.choice(FEMALE_NAMES)
            pair = rng.sample(male_names, 2)
            changer_name, stayer_name = pair

            if rng.random() < 0.5:
                name_a, name_b = stayer_name, changer_name
            else:
                name_a, name_b = changer_name, stayer_name

            changer_phrase = rng.choice(changer_phrases)
            stayer_phrase = rng.choice(stayer_phrases_to_use)
            location = rng.choice(locations)
            topic = rng.choice(topics)
            struct, role_order, s3_variant = structural_combos[
                i % len(structural_combos)
            ]

            prompt = build_prompt(
                speaker, name_a, name_b, location,
                changer_name, changer_phrase, stayer_name, stayer_phrase,
                struct, role_order, s3_variant, topic, rng
            )

            antecedent = get_antecedent_label(
                condition_name, stayer_name, changer_name
            )

            samples.append({
                "split": split,
                "prompt": prompt,
                "condition": condition_name,
                "structure": struct,
                "role_order": role_order,
                "s3_variant": s3_variant,
                "antecedent_label": antecedent,
            })

    return samples


def generate_gen_samples(target_per_condition, rng):
    """
    Generate samples for generalisation conditions.
    No train/test split — all go into 'generalisation'.
    Uses merged name/location/topic pools for maximum variety.
    """
    samples = []

    all_male = MALE_NAMES["train"] + MALE_NAMES["test"]
    all_locations = LOCATIONS["train"] + LOCATIONS["test"]
    all_topics = TOPICS["train"] + TOPICS["test"]
    all_stayer = STAYER_PHRASES["train"] + STAYER_PHRASES["test"]

    structural_combos = list(product(STRUCTURES, ROLE_ORDERS, S3_VARIANTS))

    for condition_name, condition_phrases in GEN_CONDITIONS.items():
        all_changer = condition_phrases["train"] + condition_phrases["test"]

        for i in range(target_per_condition):
            speaker = rng.choice(FEMALE_NAMES)
            pair = rng.sample(all_male, 2)
            changer_name, stayer_name = pair

            if rng.random() < 0.5:
                name_a, name_b = stayer_name, changer_name
            else:
                name_a, name_b = changer_name, stayer_name

            changer_phrase = rng.choice(all_changer)
            stayer_phrase = rng.choice(all_stayer)
            location = rng.choice(all_locations)
            topic = rng.choice(all_topics)
            struct, role_order, s3_variant = structural_combos[
                i % len(structural_combos)
            ]

            prompt = build_prompt(
                speaker, name_a, name_b, location,
                changer_name, changer_phrase, stayer_name, stayer_phrase,
                struct, role_order, s3_variant, topic, rng
            )

            antecedent = get_antecedent_label(
                condition_name, stayer_name, changer_name
            )

            samples.append({
                "split": "generalisation",
                "prompt": prompt,
                "condition": condition_name,
                "structure": struct,
                "role_order": role_order,
                "s3_variant": s3_variant,
                "antecedent_label": antecedent,
            })

    return samples


def generate_dataset(target_total=2400, seed=42):
    rng = random.Random(seed)

    n_conditions = len(POLE_CONDITIONS) + len(GEN_CONDITIONS)  # 5
    per_condition = target_total // n_conditions                # 480

    # Pole conditions: 80/20 lexical train/test split
    pole_train = (int(per_condition * 0.8) // 16) * 16  # 384
    pole_test  = (int(per_condition * 0.2) // 16) * 16  # 96

    # Generalisation conditions: all samples, no split
    gen_count = (per_condition // 16) * 16  # 480

    actual_total = (
        (pole_train + pole_test) * len(POLE_CONDITIONS)
        + gen_count * len(GEN_CONDITIONS)
    )

    print(f"Target: {target_total} → Actual: {actual_total}")
    print(f"  Pole conditions (explicit_leave, baseline):")
    print(f"    Train: {pole_train}/cond, Test: {pole_test}/cond")
    print(f"  generalisation conditions (implied_leave, implied_cancel, disengaged):")
    print(f"    All: {gen_count}/cond (no split)")

    train_poles = generate_pole_samples("train", pole_train, rng)
    test_poles  = generate_pole_samples("test",  pole_test,  rng)
    gen_samples = generate_gen_samples(gen_count, rng)

    all_samples = train_poles + test_poles + gen_samples
    rng.shuffle(all_samples)
    return all_samples


# ═══════════════════════════════════════════════════════════════════
# 14. VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_dataset(samples):
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    # Check lexical splits (no overlap between train/test)
    issues = []
    for comp_name, comp_data in [
        ("male_names",     MALE_NAMES),
        ("locations",      LOCATIONS),
        ("topics",         TOPICS),
        ("stayer_phrases", STAYER_PHRASES),
        ("explicit_leave", EXPLICIT_LEAVE),
        ("implied_leave",  IMPLIED_LEAVE),
        ("implied_cancel", IMPLIED_CANCEL),
        ("disengaged",     DISENGAGED),
    ]:
        train_set = set(comp_data["train"])
        test_set  = set(comp_data["test"])
        overlap   = train_set & test_set
        status    = "PASS" if len(overlap) == 0 else "FAIL"
        if status == "FAIL":
            issues.append(comp_name)
        print(f"  {comp_name:20s}: {len(train_set):2d} train / {len(test_set):2d} test / overlap {len(overlap)} → {status}")

    # Count balance
    print()
    counts = defaultdict(lambda: defaultdict(int))
    for s in samples:
        counts[s["split"]][s["condition"]] += 1
        counts[s["split"]]["total"] += 1

    all_conditions = sorted(set(s["condition"] for s in samples))

    for split in ["train", "test", "generalisation"]:
        if counts[split]["total"] == 0:
            continue
        print(f"  {split.upper()}: {counts[split]['total']} total")
        for cond in all_conditions:
            if counts[split][cond] > 0:
                print(f"    {cond:20s}: {counts[split][cond]}")

    # Verify no generalisation conditions leak into train/test
    print()
    for s in samples:
        if s["condition"] in GEN_CONDITIONS and s["split"] != "generalisation":
            issues.append(f"{s['condition']} in split {s['split']}")
            break
        if s["condition"] in POLE_CONDITIONS and s["split"] == "generalisation":
            issues.append(f"{s['condition']} in generalisation")
            break

    # Verify baseline uses stayer-like phrases (no separate pool)
    baseline_samples = [s for s in samples if s["condition"] == "baseline"]
    if baseline_samples:
        print(f"  Baseline sample check (should use stayer phrases):")
        sample = baseline_samples[0]
        print(f"    {sample['prompt'][:120]}...")

    print()
    if not issues:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILED: {', '.join(str(i) for i in issues)}")

    return len(issues) == 0


# ═══════════════════════════════════════════════════════════════════
# 15. GENERATE AND EXPORT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT = "RAID.csv"
    TARGET = 2400
    SEED   = 42

    samples = generate_dataset(target_total=TARGET, seed=SEED)
    ok = validate_dataset(samples)

    fieldnames = [
        "split", "prompt", "condition", "structure",
        "role_order", "s3_variant", "antecedent_label",
    ]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)

    print(f"\nDone! {len(samples)} samples → {OUTPUT}")