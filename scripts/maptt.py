import json

def map_test_types(json_filepath="data/assessments_raw.json", output_filepath=None):
    type_map = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations"
    }
    
    print(f"Loading JSON from {json_filepath}...")
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Mapping test types for {len(data)} items...")
    
    for item in data:
        test_type = item.get("test_types", "")
        if test_type:
            mapped_types = [type_map.get(letter, letter) for letter in test_type if letter in type_map]
            item["test_types"] = mapped_types if mapped_types else []
    
    output_path = output_filepath or json_filepath
    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully mapped test types in {len(data)} items")
    print(f"Saved to {output_path}")
    return data

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/assessments_raw.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    map_test_types(input_file, output_file)
