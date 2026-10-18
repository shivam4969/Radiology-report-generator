import sys, json
print(json.dumps({"event":"ready","backend":"torch","model":"fake/vlm","device":"cpu"}), flush=True)
for line in sys.stdin:
    r=json.loads(line); print(json.dumps({"id":r["id"],"text":"The right lower lung zone shows a dense opacity obscuring the hemidiaphragm. The left lung is clear. Heart size is normal.\n\nImpression: Right lower lobe consolidation, likely pneumonia.","ms":12,"model":"fake/vlm"}), flush=True)
