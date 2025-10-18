import json, datetime
from pathlib import Path

class Logger:
    def __init__(self, exp_name, out_dir="runs"):
        self.exp_name = exp_name
        self.start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_dir = Path(out_dir) / exp_name / self.start_time
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.out_dir / "train_log.jsonl"

    def log(self, **kwargs):
        entry = {"time": datetime.datetime.now().isoformat(), **kwargs}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(entry)

    def save_meta(self, meta):
        with open(self.out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
