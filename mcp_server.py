import sys
import json
from client import ToTMCTS

mcts = ToTMCTS()

def handle_call(name, arguments):
    if name == "expand":
        pid = arguments.get("parent_id", 0)
        cands = arguments["candidates"]
        cid = mcts.expand(pid, cands)
        return {"first_child": cid, "total_nodes": len(mcts.tree)}
    elif name == "backprop":
        nid = arguments["node_id"]
        rew = arguments["reward"]
        mcts.backpropagate(nid, rew)
        return {"status": "ok"}
    elif name == "select":
        return {"selected": mcts.select(0)}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
