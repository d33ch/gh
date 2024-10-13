import json
from difflib import SequenceMatcher


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def similarity_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def create_mapping(tracks_data, market_tracks):
    mapping = {}
    for market_track in market_tracks:
        best_match = None
        best_ratio = 0
        for track in tracks_data:
            ratio = similarity_ratio(market_track, track["trackName"])
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = track

        if best_ratio >= 0.8:
            mapping[market_track] = {
                "matchedTrackName": best_match["trackName"],
                "trackId": best_match["trackId"],
                "trackCode": best_match["trackCode"],
                "owningAuthorityCode": best_match["owningAuthorityCode"],
                "active": best_match["active"],
                "similarityScore": best_ratio,
            }
        else:
            mapping[market_track] = None

    return mapping


tracks_data = load_json_file("tracks.json")
market_tracks = load_json_file("market_unique_tracks_2017.json")

mapping = create_mapping(tracks_data, market_tracks)

print("Mapping:")
for market_track, matched_track in mapping.items():
    print(f"{market_track}: {matched_track}")

unmatched = [track for track, match in mapping.items() if match is None]
print("\nUnmatched tracks:")
for track in unmatched:
    print(track)

with open("track_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)
