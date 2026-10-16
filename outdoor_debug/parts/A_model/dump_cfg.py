import sys, json
sys.path.insert(0, "/home/wg19671/.claude/jobs/a0d391cb/tmp/od")
import train, yaml
for n in sys.argv[1:]:
    c = train.load_config(__import__("pathlib").Path(f"configs/experiments/{n}.yaml"))
    print("=====", n)
    print("model:", json.dumps(c["model"], indent=1, default=str))
    print("loss:", json.dumps(c.get("loss"), indent=1, default=str))
    print("data keys:", {k:v for k,v in c["data"].items() if k!="sources"})
    print("sources:", [(s["type"], s.get("max_depth"), s.get("silog_max_depth"), s.get("vkitti_max_depth"), s.get("metric_supervision")) for s in c["data"]["sources"]])
    print("training:", json.dumps(c.get("training"), indent=1, default=str))
